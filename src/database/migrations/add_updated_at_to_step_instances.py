#!/usr/bin/env python3
"""
数据库迁移脚本：为step_instances表添加updated_at字段
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.database.database_config import DatabaseConfig
from sqlalchemy import text
from datetime import datetime

def add_updated_at_to_step_instances():
    """为step_instances表添加updated_at字段"""
    
    db_config = DatabaseConfig()
    
    # SQL语句
    add_column_sql = text("""
    ALTER TABLE workflow.step_instances 
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
    """)
    
    update_existing_sql = text("""
    UPDATE workflow.step_instances 
    SET updated_at = COALESCE(completed_at, started_at, CURRENT_TIMESTAMP)
    WHERE updated_at IS NULL;
    """)
    
    add_comment_sql = text("""
    COMMENT ON COLUMN workflow.step_instances.updated_at IS '更新时间';
    """)
    
    try:
        with db_config.get_session() as session:
            print("🚀 开始为step_instances表添加updated_at字段...")
            
            # 添加字段
            session.execute(add_column_sql)
            print("✅ 添加updated_at字段成功")
            
            # 更新现有记录
            result = session.execute(update_existing_sql)
            print(f"✅ 更新了 {result.rowcount} 条现有记录")
            
            # 添加注释
            session.execute(add_comment_sql)
            print("✅ 添加字段注释成功")
            
            # 提交更改
            session.commit()
            print("✅ 数据库迁移完成")
            
    except Exception as e:
        print(f"❌ 迁移失败: {str(e)}")
        raise e

if __name__ == "__main__":
    add_updated_at_to_step_instances() 