#!/usr/bin/env python3
"""
数据库连接测试脚本

测试PostgreSQL数据库连接是否正常
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.database.database_config import db_config
from sqlalchemy import text

def test_basic_connection():
    """测试基本数据库连接"""
    print("🔍 测试基本数据库连接...")
    
    try:
        with db_config.engine.connect() as connection:
            result = connection.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            if row and row[0] == 1:
                print("✅ 基本连接测试成功")
                return True
            else:
                print("❌ 基本连接测试失败")
                return False
    except Exception as e:
        print(f"❌ 基本连接测试失败: {e}")
        return False

def test_database_info():
    """测试数据库信息查询"""
    print("🔍 查询数据库信息...")
    
    try:
        with db_config.engine.connect() as connection:
            # PostgreSQL version
            result = connection.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"📊 PostgreSQL版本: {version.split(',')[0]}")
            
            # Current database
            result = connection.execute(text("SELECT current_database()"))
            database = result.fetchone()[0]
            print(f"📊 当前数据库: {database}")
            
            # Current user
            result = connection.execute(text("SELECT current_user"))
            user = result.fetchone()[0]
            print(f"📊 当前用户: {user}")
            
            return True
    except Exception as e:
        print(f"❌ 数据库信息查询失败: {e}")
        return False

def test_table_operations():
    """测试表操作"""
    print("🔍 测试表操作...")
    
    try:
        with db_config.engine.connect() as connection:
            # Create test table
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS test_table (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            connection.commit()
            print("✅ 测试表创建成功")
            
            # Insert test data
            connection.execute(text("""
                INSERT INTO test_table (name) VALUES ('test_connection')
            """))
            connection.commit()
            print("✅ 测试数据插入成功")
            
            # Query test data
            result = connection.execute(text("""
                SELECT id, name, created_at FROM test_table WHERE name = 'test_connection'
            """))
            row = result.fetchone()
            if row:
                print(f"✅ 测试数据查询成功: ID={row[0]}, Name={row[1]}")
            
            # Clean up
            connection.execute(text("DROP TABLE IF EXISTS test_table"))
            connection.commit()
            print("✅ 测试表清理完成")
            
            return True
    except Exception as e:
        print(f"❌ 表操作测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 PostgreSQL数据库连接测试")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Display configuration
    print(f"📊 数据库URL: {db_config.database_url}")
    print()
    
    # Run tests
    tests = [
        ("基本连接测试", test_basic_connection),
        ("数据库信息查询", test_database_info),
        ("表操作测试", test_table_operations)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
        print()
    
    # Summary
    print("=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！数据库连接正常")
        return True
    else:
        print("❌ 部分测试失败，请检查数据库配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 