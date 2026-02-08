
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from data_fetch.task.update_min5_task import UpdateTask

class TestUpdateTask(unittest.TestCase):

    def setUp(self):
        self.task = UpdateTask()

    @patch('data_fetch.task.update_min5_task.BaoInterface')
    @patch('data_fetch.task.update_min5_task.ProcessPoolExecutor')
    @patch('data_fetch.task.update_min5_task.SqlOp')
    def test_logic(self, MockSqlOp, MockExecutor, MockBaoInterface):
        # Setup Mocks
        mock_sql_op = MockSqlOp.return_value
        # Use a real UpdateTask instance, but with the mocked sql_op injected if needed, 
        # or just let the constructor create the mock (since we patched the class).
        # The constructor of UpdateTask calls SqlOp(), so self.task.sql_op is already a mock 
        # (but a different instance if we created self.task before patching).
        # So we should re-instantiate UpdateTask inside the test or patch where it's instantiated.
        
        # Let's re-instantiate to use the patched SqlOp class
        task = UpdateTask()
        task.sql_op = mock_sql_op # Ensure we control the instance
        
        # Mock BaoInterface context manager
        mock_bi_instance = MockBaoInterface.return_value
        mock_bi_instance.__enter__.return_value = mock_bi_instance
        
        # Mock get_trade_dates
        mock_bi_instance.get_trade_dates.return_value = pd.DataFrame({
            'calendar_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'is_trading_day': [1, 1, 1, 1, 1]
        })
        
        # Mock get_stock_list
        # Return a few test codes
        # Code 1: sh.600000 - Needs Update (New Data)
        # Code 2: sz.000001 - Needs Front Update (Old Data)
        # Code 3: sz.300001 - Needs Both
        # Code 4: sh.600001 - Up to date
        # Code 5: sh.600002 - No data in DB
        
        mock_bi_instance.get_stock_list.return_value = pd.DataFrame({
            'code': ['sh.600000', 'sz.000001', 'sz.300001', 'sh.600001', 'sh.600002'],
            'code_name': ['浦发银行', '平安银行', '特锐德', '邯郸钢铁', '齐鲁石化']
        })
        
        # Mock SqlOp responses
        # get_max_date_for_codes
        mock_sql_op.get_max_date_for_codes.return_value = {
            'sh.600000': '2023-01-03', # Needs update to 2023-01-05
            'sz.000001': '2023-01-05', # Max is current, but maybe min is old?
            'sz.300001': '2023-01-03', # Needs update to 2023-01-05 AND maybe front
            'sh.600001': '2023-01-05', # Up to date
            # sh.600002 missing (no data)
        }
        
        # get_min_date_for_codes
        mock_sql_op.get_min_date_for_codes.return_value = {
            'sh.600000': '2023-01-01', # Start matches
            'sz.000001': '2023-01-03', # Start is 2023-01-01, so needs front update
            'sz.300001': '2023-01-03', # Start is 2023-01-01, needs front update
            'sh.600001': '2023-01-01', # Start matches
            # sh.600002 missing
        }

        # Mock Executor
        mock_executor_instance = MockExecutor.return_value
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        
        # We want to capture the calls to submit
        # submit(fn, code, start, end, adjustflag)
        
        # Run the task
        start_date = '2023-01-01'
        end_date = '2023-01-05'
        task.run_init_mintues5_task(start_date=start_date, end_date=end_date, adjustflag='2', batch_size=10)
        
        # Verify calls
        # We expect submit to be called for:
        # sh.600000: start='2023-01-03' (Last max), end='2023-01-05'
        # sz.000001: start='2023-01-01', end='2023-01-03' (First min)
        # sz.300001: start='2023-01-01', end='2023-01-05' (Both gaps -> full fetch logic)
        # sh.600001: Should NOT be submitted (Fully covered)
        # sh.600002: start='2023-01-01', end='2023-01-05' (No data)
        
        submitted_calls = {}
        for call in mock_executor_instance.submit.call_args_list:
            args = call.args
            # args[0] is the function 'fetch_data_task'
            code = args[1]
            s_date = args[2]
            e_date = args[3]
            adj = args[4]
            submitted_calls[code] = (s_date, e_date, adj)
            
        print("\nSubmitted Tasks:")
        for k, v in submitted_calls.items():
            print(f"{k}: {v}")

        # Assertions
        
        # Case 1: Incremental Update (New Data)
        # Logic says: if has_new and not has_old -> start = max_date
        self.assertIn('sh.600000', submitted_calls)
        self.assertEqual(submitted_calls['sh.600000'], ('2023-01-03', '2023-01-05', '2'))
        
        # Case 2: Front Update (Old Data)
        # Logic says: if has_old and not has_new -> end = min_date
        self.assertIn('sz.000001', submitted_calls)
        self.assertEqual(submitted_calls['sz.000001'], ('2023-01-01', '2023-01-03', '2'))
        
        # Case 3: Both Gaps
        # Logic says: if has_new and has_old -> pass (fetch full range)
        self.assertIn('sz.300001', submitted_calls)
        self.assertEqual(submitted_calls['sz.300001'], ('2023-01-01', '2023-01-05', '2'))
        
        # Case 4: No Data
        self.assertIn('sh.600002', submitted_calls)
        self.assertEqual(submitted_calls['sh.600002'], ('2023-01-01', '2023-01-05', '2'))
        
        # Case 5: No Update Needed
        self.assertNotIn('sh.600001', submitted_calls)

if __name__ == '__main__':
    unittest.main()
