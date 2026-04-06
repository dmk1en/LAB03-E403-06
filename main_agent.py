import os
import sys
import time
from dotenv import load_dotenv
from src.agent.agent import ReActAgent
from src.tools.ecommerce_tools import ECOMMERCE_TOOLS_SPEC
from src.core.openai_provider import OpenAIProvider
# from src.core.local_provider import LocalProvider
# from src.core.gemini_provider import GeminiProvider
from src.telemetry.metrics import tracker


def get_provider():
    load_dotenv()
    
    provider_name = os.getenv("DEFAULT_PROVIDER", "openai").lower()
    
    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ Lỗi: OPENAI_API_KEY chưa được thiết lập trong file .env")
            sys.exit(1)
        return OpenAIProvider(model_name="gpt-4o-mini", api_key=api_key)
        
    elif provider_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ Lỗi: GEMINI_API_KEY chưa được thiết lập trong file .env")
            sys.exit(1)
        return GeminiProvider(model_name="gemini-1.5-flash", api_key=api_key)
        
    elif provider_name == "local":
        model_path = os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf")
        if not os.path.exists(model_path):
            print(f"❌ Lỗi: Không tìm thấy model cục bộ tại {model_path}")
            sys.exit(1)
        return LocalProvider(model_path=model_path)
    
    else:
        print(f"❌ Lỗi: Provider '{provider_name}' không được hỗ trợ.")
        sys.exit(1)

def run_tests():
    print("🚀 Đang khởi tạo ReAct Agent v1...")
    provider = get_provider()
    agent = ReActAgent(llm=provider, tools=ECOMMERCE_TOOLS_SPEC, max_steps=7)
    
    # 5 Kịch bản (Test Cases) phủ đủ độ khó từ Dễ đến Ảo Giác
    test_cases = [
        {
            "id": 1,
            "level": "Dễ",
            "desc": "Truy vấn 1 công cụ (Single Tool)",
            "query": "Kiểm tra giúp tôi xem kho còn điện thoại iPhone không?"
        },
        {
            "id": 2,
            "level": "Trung bình",
            "desc": "Kết hợp 2 công cụ rời rạc",
            "query": "Tôi có mã giảm giá TET. Nếu kiện hàng của tôi nặng 2kg và muốn ship về HCM, thì mã giảm giá là mấy phần trăm và phí giao hàng là bao nhiêu tiền?"
        },
        {
            "id": 3,
            "level": "Khó",
            "desc": "Luồng mua hàng hoàn chỉnh (Full Flow)",
            "query": "Hàng Macbook còn không? Nếu còn thì tính tổng tiền hóa đơn cho tôi biết: Mua 2 cái Macbook, tổng nặng 4kg, chuyển về Hà Nội. Gắn thêm mã giảm WINNER. (Lưu ý giá 1 chiếc là 30 triệu VND)."
        },
        {
            "id": 4,
            "level": "Bẫy Ảo Giác",
            "desc": "Sản phẩm không có trong quy định",
            "query": "Tôi muốn mua đôi giày Sneaker Nike nặng 1kg ship về Hà Nội. Check kho cho tôi."
        },
        {
            "id": 5,
            "level": "Ngoại lệ",
            "desc": "Hội thoại thông thường không cần Tool",
            "query": "Chào em, dạo này em bán hàng có tốt không? Có hay bị kẹt đơn không?"
        },
        {
            "id": 6,
            "level": "Tính toán",
            "desc": "Test Tool Tính tổng tiền mới",
            "query": "Tính giúp tôi tổng tiền nếu tôi mua 5 chiếc điện thoại iPhone, giá mỗi cái là 25000000 VND."
        },
        {
            "id": 7,
            "level": "API Nâng cao",
            "desc": "Kết nối API Internet lấy Tỷ giá thực tế",
            "query": "Khách nước ngoài qua Việt Nam ngỏ ý muốn mua 1 cái Macbook giá 35 triệu VNĐ. Hãy quy đổi 35 triệu VNĐ đó ra tiền đô la USD báo cho khách giúp tôi nhé."
        }
    ]

    while True:
        print("\n" + "="*60)
        print("📋 MENU CHỌN TEST CASE ".center(60, "="))
        for tc in test_cases:
            print(f"[{tc['id']}] Khúc {tc['level']} - {tc['desc']}")
        print("[0] Chạy TẤT CẢ 7 Test Cases lần lượt")
        print("[q] Thoát chương trình")
        print("="*60)
        
        choice = input("👉 Mời nhập ID (0-7, hoặc 'q'): ").strip().lower()
        
        if choice == 'q' or choice == 'quit':
            print("👋 Đã thoát test.")
            break
            
        try:
            choice_int = int(choice)
        except ValueError:
            print("❌ Vui lòng nhập số hợp lệ.")
            continue
            
        if choice_int == 0:
            target_cases = test_cases
        elif 1 <= choice_int <= 7:
            target_cases = [test_cases[choice_int - 1]]
        else:
            print("❌ Không có test case này. Vui lòng nhập từ 0-7.")
            continue

        for tc in target_cases:
            print("\n" + "="*80)
            print(f"🔥 ĐANG CHẠY TEST CASE {tc['id']} | Độ khó: [{tc['level']}] - {tc['desc']}")
            print(f"👤 USER: {tc['query']}")
            print("="*80)

            # Reset per-run token tracking
            tracker.reset_session()

            # Khởi chạy Agent thay vì phải reset history (vì logic lab 3 mỗi run là 1 query mới)
            final_response = agent.run(tc['query'])

            print("\n" + "*"*80)
            print(f"🏆 FINAL ANSWER {tc['id']}:\n {final_response}")

            # Token & cost summary
            summary = tracker.get_session_summary()
            if summary:
                print(f"\n📊 TELEMETRY DASHBOARD (Test Case {tc['id']}):")
                print(f"   Độ trễ trung bình (P50)        : {summary['total_latency_ms']} ms")
                print(f"   Độ trễ tối đa (P99)           : {summary['total_latency_ms']} ms (single test case)")
                print(f"   Số token trung bình mỗi task  : {summary['total_tokens']}")
                print(f"   Tổng chi phí test             : ${summary['total_cost_usd']:.6f}")
            print("*"*80 + "\n")
            
            if choice_int == 0:  # Pause một chút nếu chạy toàn bộ 5 case để API đỡ nóng
                import time
                time.sleep(1.5)

if __name__ == "__main__":
    run_tests()
