import sys
import os
sys.path.insert(0, os.getcwd())

from stock.tasks.migrate_concept import ConceptMigrationTask

def update_concepts():
    print("="*70)
    print("🛠️  独立任务：更新概念板块映射")
    print("="*70)
    
    task = ConceptMigrationTask()
    try:
        task.run()
    finally:
        task.close()

if __name__ == "__main__":
    update_concepts()
