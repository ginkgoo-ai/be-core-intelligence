#!/usr/bin/env python3
"""
数据库迁移：添加层级化步骤支持
为 step_instances 表添加 parent_step_id 和 step_type 字段
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.database_config import db_config

def run_migration():
    """执行数据库迁移"""
    
    # 获取数据库引擎
    engine = db_config.engine
    
    print("开始执行层级化步骤迁移...")
    
    with engine.connect() as connection:
        # 开始事务
        trans = connection.begin()
        
        try:
            # 1. 添加 parent_step_id 字段
            print("1. 添加 parent_step_id 字段...")
            connection.execute(text("""
                ALTER TABLE workflow.step_instances 
                ADD COLUMN parent_step_id VARCHAR(50) NULL
            """))
            
            # 2. 添加 step_type 字段
            print("2. 添加 step_type 字段...")
            connection.execute(text("""
                ALTER TABLE workflow.step_instances 
                ADD COLUMN step_type VARCHAR(10) DEFAULT 'MAIN'
            """))
            
            # 3. 添加字段注释
            print("3. 添加字段注释...")
            connection.execute(text("""
                COMMENT ON COLUMN workflow.step_instances.parent_step_id 
                IS '父步骤ID(外键,用于层级化工作流)'
            """))
            
            connection.execute(text("""
                COMMENT ON COLUMN workflow.step_instances.step_type 
                IS '步骤类型(MAIN=主步骤,SUB=子步骤)'
            """))
            
            # 4. 添加外键约束
            print("4. 添加父子关系外键约束...")
            connection.execute(text("""
                ALTER TABLE workflow.step_instances 
                ADD CONSTRAINT fk_step_parent 
                FOREIGN KEY (parent_step_id) 
                REFERENCES workflow.step_instances(step_instance_id) 
                ON DELETE CASCADE
            """))
            
            # 5. 添加索引提高查询性能
            print("5. 添加索引...")
            connection.execute(text("""
                CREATE INDEX idx_step_parent_id 
                ON workflow.step_instances(parent_step_id)
            """))
            
            connection.execute(text("""
                CREATE INDEX idx_step_type 
                ON workflow.step_instances(step_type)
            """))
            
            connection.execute(text("""
                CREATE INDEX idx_workflow_step_type 
                ON workflow.step_instances(workflow_instance_id, step_type)
            """))
            
            # 提交事务
            trans.commit()
            print("✅ 层级化步骤迁移完成！")
            
        except Exception as e:
            # 回滚事务
            trans.rollback()
            print(f"❌ 迁移失败: {str(e)}")
            raise e

def rollback_migration():
    """回滚数据库迁移"""
    
    engine = db_config.engine
    
    print("开始回滚层级化步骤迁移...")
    
    with engine.connect() as connection:
        trans = connection.begin()
        
        try:
            # 删除索引
            print("1. 删除索引...")
            connection.execute(text("DROP INDEX IF EXISTS idx_workflow_step_type ON workflow.step_instances"))
            connection.execute(text("DROP INDEX IF EXISTS idx_step_type ON workflow.step_instances"))
            connection.execute(text("DROP INDEX IF EXISTS idx_step_parent_id ON workflow.step_instances"))
            
            # 删除外键约束
            print("2. 删除外键约束...")
            connection.execute(text("ALTER TABLE workflow.step_instances DROP FOREIGN KEY IF EXISTS fk_step_parent"))
            
            # 删除字段
            print("3. 删除字段...")
            connection.execute(text("ALTER TABLE workflow.step_instances DROP COLUMN IF EXISTS step_type"))
            connection.execute(text("ALTER TABLE workflow.step_instances DROP COLUMN IF EXISTS parent_step_id"))
            
            trans.commit()
            print("✅ 层级化步骤迁移回滚完成！")
            
        except Exception as e:
            trans.rollback()
            print(f"❌ 回滚失败: {str(e)}")
            raise e

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="层级化步骤数据库迁移")
    parser.add_argument("--rollback", action="store_true", help="回滚迁移")
    
    args = parser.parse_args()
    
    if args.rollback:
        rollback_migration()
    else:
        run_migration() 