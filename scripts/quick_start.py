#!/usr/bin/env python3
"""
快速启动脚本

自动完成环境配置、数据库初始化和服务启动
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description}成功")
            if result.stdout.strip():
                print(f"   输出: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description}失败")
            if result.stderr.strip():
                print(f"   错误: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description}失败: {e}")
        return False

def check_postgresql():
    """检查PostgreSQL是否运行"""
    print("🔍 检查PostgreSQL服务...")
    
    # Try to connect using psql
    command = 'psql -h 127.0.0.1 -p 5433 -U postgres -d postgres -c "SELECT 1;" 2>/dev/null'
    result = subprocess.run(command, shell=True, capture_output=True)
    
    if result.returncode == 0:
        print("✅ PostgreSQL服务正在运行")
        return True
    else:
        print("❌ PostgreSQL服务未运行或连接失败")
        print("   请确保PostgreSQL服务已启动并配置正确")
        print("   默认配置: 主机=127.0.0.1, 端口=5433, 用户=postgres")
        return False

def setup_environment():
    """设置环境"""
    print("🚀 签证自动填表工作流系统 - 快速启动")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python版本过低，需要Python 3.8+")
        return False
    
    print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check PostgreSQL
    if not check_postgresql():
        return False
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "安装依赖包"):
        return False
    
    # Setup environment variables
    if not run_command("python scripts/setup_env.py", "配置环境变量"):
        print("⚠️  环境变量配置可能失败，请手动创建.env文件")
    
    # Test database connection
    if not run_command("python scripts/test_database.py", "测试数据库连接"):
        return False
    
    # Initialize database
    if not run_command("python scripts/init_database.py", "初始化数据库"):
        return False
    
    return True

def start_application():
    """启动应用"""
    print("\n🚀 启动应用服务...")
    print("=" * 60)
    
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        print("📊 服务配置:")
        print(f"   主机: {os.getenv('APP_HOST', '0.0.0.0')}")
        print(f"   端口: {os.getenv('APP_PORT', '8000')}")
        print(f"   数据库: PostgreSQL")
        print(f"   调试模式: {os.getenv('APP_RELOAD', 'true')}")
        
        print("\n🌐 API文档地址:")
        print("   Swagger UI: http://localhost:8000/docs")
        print("   ReDoc: http://localhost:8000/redoc")
        print("   健康检查: http://localhost:8000/health")
        
        print("\n🔄 启动中...")
        print("   按 Ctrl+C 停止服务")
        print("=" * 60)
        
        # Start the application
        os.system("python main.py")
        
    except KeyboardInterrupt:
        print("\n\n🛑 服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")

def main():
    """主函数"""
    try:
        # Setup environment
        if setup_environment():
            print("\n🎉 环境配置完成！")
            
            # Ask user if they want to start the application
            response = input("\n是否立即启动应用服务? (Y/n): ")
            if response.lower() in ['', 'y', 'yes']:
                start_application()
            else:
                print("\n📋 手动启动命令:")
                print("   python main.py")
        else:
            print("\n❌ 环境配置失败，请检查错误信息")
            return False
            
    except KeyboardInterrupt:
        print("\n\n🛑 操作已取消")
    except Exception as e:
        print(f"\n❌ 启动脚本失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 