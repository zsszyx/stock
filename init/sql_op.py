
import os
import sqlite3


def reset_database(db_path="stock_data.db"):
    """
    清除数据库并重建。
    :param db_path: 数据库文件路径，默认为 'stock_data.db'
    """
    # 删除现有数据库文件
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"数据库 {db_path} 已删除。")

    # 创建新的数据库文件
    conn = sqlite3.connect(db_path)
    conn.close()
    print(f"数据库 {db_path} 已重建。")

def show_database_info(db_path="stock_data.db"):
    """
    显示数据库存储大小以及表数量。
    :param db_path: 数据库文件路径，默认为 'stock_data.db'
    """
    import os
    import sqlite3

    # 检查数据库文件是否存在
    if not os.path.exists(db_path):
        print(f"数据库文件 {db_path} 不存在！")
        return

    # 获取数据库文件大小
    db_size = os.path.getsize(db_path) / (1024 * 1024)  # 转换为 MB
    print(f"数据库文件大小: {db_size:.2f} MB")

    # 连接 SQLite 数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 获取表数量
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
    table_count = cursor.fetchone()[0]
    print(f"数据库中表的数量: {table_count}")

if __name__ == "__main__":
    # 示例：重置数据库并显示信息
    reset_database()
    show_database_info()
    # 你可以在这里添加更多的测试代码或功能调用