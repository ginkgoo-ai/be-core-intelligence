#!/usr/bin/env python3
"""
数据库初始化脚本

用于初始化PostgreSQL数据库表结构
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.database.database_config import init_database, db_config

def main():
    """主函数"""
    print("🚀 开始初始化数据库...")
    
    # Load environment variables
    load_dotenv()
    
    # Display configuration
    print(f"📊 数据库URL: {db_config.database_url}")
    
    try:
        # Test connection first
        if db_config.test_connection():
            print("✅ 数据库连接测试成功")
        else:
            print("❌ 数据库连接测试失败")
            return False
        
        # Initialize database
        init_database()
        print("🎉 数据库初始化完成！")
        return True
        
    except Exception as e:
        print(f"❌ 数据库初始化失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 