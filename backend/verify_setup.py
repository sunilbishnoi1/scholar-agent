"""
System Diagnostic & Verification Tool for Scholar Agent.
Run this script to verify all API keys, external services, and system components.

Usage:
    python verify_setup.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from project root .env or backend .env
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check_brevo():
    print_header("1. Brevo (Sendinblue) Email Service Check")
    api_key = os.environ.get("BREVO_API_KEY", "").strip().strip('"').strip("'")
    sender_email = os.environ.get("BREVO_SENDER_EMAIL", "sunilbishnoi7205@gmail.com").strip().strip('"').strip("'")
    sender_name = os.environ.get("BREVO_SENDER_NAME", "Scholar AI Agent").strip()

    if not api_key or api_key in ["your_actual_api_key_here", "your-brevo-api-key", "your-brevo-api-key-here"]:
        print("  [!] BREVO_API_KEY: Not set or placeholder. Email notifications will be skipped.")
        print("      (This is optional. You can still use the web UI to view and download reports).")
        return

    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"  [i] API Key: {masked_key}")
    print(f"  [i] Sender Email: {sender_email}")
    print(f"  [i] Sender Name: {sender_name}")

    try:
        import sib_api_v3_sdk
        from sib_api_v3_sdk.rest import ApiException
        
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key["api-key"] = api_key
        api_client = sib_api_v3_sdk.ApiClient(configuration)
        
        # Test account API to verify API key permissions
        account_api = sib_api_v3_sdk.AccountApi(api_client)
        account_info = account_api.get_account()
        
        print("  [OK] Brevo API Key is ACTIVE and VALID!")
        print(f"       Account Email: {account_info.email}")
        print(f"       Company Name: {getattr(account_info, 'company_name', 'N/A')}")
        if hasattr(account_info, "plan"):
            print(f"       Plan: {[p.type for p in account_info.plan] if account_info.plan else 'Free'}")
    except ApiException as e:
        if getattr(e, "status", None) == 401:
            print(f"  [ERROR] Brevo API Key Unauthorized (401): {e.body if hasattr(e, 'body') else e}")
            print("          Tip: Make sure the key was generated in Brevo under SMTP & API -> API Keys,")
            print("          and that your Brevo account email is confirmed and transactional service is activated.")
        else:
            print(f"  [ERROR] Brevo API Error ({getattr(e, 'status', 'unknown')}): {e}")
    except Exception as e:
        print(f"  [ERROR] Brevo check failed: {e}")

def check_llm():
    print_header("2. LLM Provider Check")
    from agents.llm import get_llm_client
    from agents.llm.factory import get_available_providers, get_best_available_provider

    available = get_available_providers()
    print(f"  [i] Available providers from environment: {[p.value for p in available]}")
    best = get_best_available_provider()
    print(f"  [i] Selected default provider: {best.value}")

    try:
        client = get_llm_client()
        provider_name = client.get_provider_name()
        print(f"  [i] Active client provider: {provider_name}")
        
        if provider_name != "mock":
            print(f"  [~] Testing live completion with {provider_name}...")
            response = client.generate_text("Reply with exactly: 'LLM is working'", max_tokens=20)
            print(f"  [OK] LLM response received: {response.strip()[:60]}")
        else:
            print("  [!] Using Mock LLM provider. Set GROQ_API_KEY or GEMINI_API_KEY in .env for real AI reasoning.")
    except Exception as e:
        print(f"  [ERROR] LLM test failed: {e}")

def check_database():
    print_header("3. Database Check")
    db_url = os.environ.get("DATABASE_URL", "sqlite:///./test.db")
    masked_db = db_url if "sqlite" in db_url else db_url.split("@")[-1] if "@" in db_url else "Postgres"
    print(f"  [i] Database target: {masked_db}")
    
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False} if "sqlite" in db_url else {}
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("  [OK] Database connection successful!")
    except Exception as e:
        print(f"  [ERROR] Database connection failed: {e}")

def check_redis():
    print_header("4. Redis / Real-Time Cache Check")
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    print(f"  [i] Configured Redis URL: {redis_url}")
    
    try:
        from cache.redis_cache import get_cache
        cache = get_cache()
        if cache and cache.is_connected:
            print("  [OK] Redis is ONLINE! (Distributed Celery queue & multi-worker Pub/Sub enabled)")
        else:
            print("  [OK] Redis is OFFLINE -> Automatic fallback ACTIVE:")
            print("       - In-memory WebSocket broadcasting: Enabled")
            print("       - Background threading execution: Enabled")
            print("       - System operates at 100% functionality in single-instance/local mode.")
    except Exception as e:
        print(f"  [i] Redis fallback active ({e})")

def check_qdrant():
    print_header("5. Vector Store (Qdrant) Check")
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    print(f"  [i] Configured Qdrant URL: {qdrant_url}")
    
    try:
        import urllib.request
        req = urllib.request.Request(f"{qdrant_url.rstrip('/')}/collections", headers={"User-Agent": "ScholarAgent/1.0"})
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                print("  [OK] Qdrant Vector Store is ONLINE! (Full RAG vector indexing enabled)")
            else:
                print(f"  [!] Qdrant returned status {response.status}")
    except Exception:
        print("  [OK] Qdrant is OFFLINE -> Semantic search falls back to multi-source academic search.")

def main():
    print("\n" + "#" * 60)
    print("      SCHOLAR AGENT SYSTEM DIAGNOSTICS & VERIFICATION")
    print("#" * 60)
    
    check_brevo()
    check_llm()
    check_database()
    check_redis()
    check_qdrant()
    
    print("\n" + "=" * 60)
    print("  Diagnostics complete!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
