#!/usr/bin/env python3
"""
环境变量设置脚本

帮助用户设置PostgreSQL数据库连接环境变量
"""

import os
from pathlib import Path

def create_env_file():
    """创建.env文件"""
    env_content = """# Database Configuration
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5433
POSTGRES_DB=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Constructed Database URL
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5433/postgres

# AI Model Configuration
MODEL_BASE_URL=https://api.openai.com/v1
MODEL_API_KEY=your_openai_api_key_here
MODEL_WORKFLOW_NAME=gpt-3.5-turbo

# Debug Settings
SQL_DEBUG=false
LOG_LEVEL=INFO

# Application Settings
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=true
"""
    
    env_file = Path(".env")
    
    if env_file.exists():
        print("⚠️  .env文件已存在")
        response = input("是否覆盖现有文件? (y/N): ")
        if response.lower() != 'y':
            print("❌ 取消操作")
            return False
    
    try:
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("✅ .env文件创建成功")
        print(f"📁 文件位置: {env_file.absolute()}")
        return True
    except Exception as e:
        print(f"❌ 创建.env文件失败: {e}")
        return False

def set_environment_variables():
    """设置环境变量"""
    env_vars = {
        "POSTGRES_HOST": "127.0.0.1",
        "POSTGRES_PORT": "5433", 
        "POSTGRES_DB": "postgres",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "postgres",
        "DATABASE_URL": "postgresql://postgres:postgres@127.0.0.1:5433/postgres"
    }
    
    print("🔧 设置环境变量...")
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   {key}={value}")
    
    print("✅ 环境变量设置完成")

def main():
    """主函数"""
    print("🚀 PostgreSQL环境配置工具")
    print("=" * 50)
    
    print("\n1. 创建.env文件")
    if create_env_file():
        print("\n2. 设置环境变量")
        set_environment_variables()
        
        print("\n🎉 配置完成！")
        print("\n📋 下一步操作:")
        print("   1. 确保PostgreSQL服务正在运行")
        print("   2. 运行: python scripts/init_database.py")
        print("   3. 启动应用: python -m src.main")
    else:
        print("\n❌ 配置失败")

if __name__ == "__main__":
    main() 