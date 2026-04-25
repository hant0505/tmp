"""
SQL Intelligence - Main Pipeline
Chạy: python main.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crewai import Crew, Task, Process
from agents.agents import create_agents
from dotenv import load_dotenv
load_dotenv()

# ── Cấu hình LLM ─────────────────────────────────────────────────────────────
# Nhóm dùng Gemini theo SDS. Đổi thành OpenAI/Anthropic tùy API key có sẵn.
def get_llm():
    """
    Chọn LLM provider. Ưu tiên theo thứ tự:
    1. Google Gemini (theo SDS của nhóm)
    2. OpenAI GPT-4
    3. Anthropic Claude
    4.Openrouter (Gemini qua OpenRouter)
    """
    from crewai import LLM

    # ✅ Option 1: OpenRouter (Gemini)
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_BASE_URL"):
        return LLM(
            model="openrouter/google/gemini-2.5-flash",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            max_tokens=2000,   # 🔥 THÊM DÒNG NÀY
            temperature=0.1    # (optional, tốt cho SQL)
        )
    # Option 1: Gemini (nhóm dùng)
    if os.getenv("GOOGLE_API_KEY"):
        return LLM(model="gemini-3-flash", api_key=os.getenv("GOOGLE_API_KEY"))

    # Option 2: OpenAI
    if os.getenv("OPENAI_API_KEY"):
        return LLM(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

    # Option 3: Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        return LLM(model="anthropic/claude-3-5-haiku-20241022", api_key=os.getenv("ANTHROPIC_API_KEY"))

    raise ValueError(
        "Chưa có API key! Set 1 trong các biến môi trường:\n"
        "  export GOOGLE_API_KEY=your_key\n"
        "  export OPENAI_API_KEY=your_key\n"
        "  export ANTHROPIC_API_KEY=your_key"
    )


# ── Tạo Tasks cho từng Agent ──────────────────────────────────────────────────
def create_tasks(user_question: str, planner, generator, executor, interpreter):

    task_plan = Task(
        description=f"""
Câu hỏi của người dùng: "{user_question}"

Nhiệm vụ:
1. Dùng tool get_database_schema để xem schema Chinook
2. Phân tích câu hỏi và tạo kế hoạch truy vấn chi tiết gồm:
   - Bảng cần dùng (và lý do)
   - Cột cần SELECT
   - Điều kiện lọc (WHERE)
   - JOIN path (bảng nào JOIN bảng nào qua key nào)
   - Phép tổng hợp: GROUP BY, ORDER BY, LIMIT
""",
        expected_output=(
            "Kế hoạch truy vấn có cấu trúc rõ ràng với: "
            "danh sách bảng, cột, JOIN path, điều kiện lọc, phép tổng hợp"
        ),
        agent=planner,
    )

    task_generate = Task(
        description="""
Dựa trên kế hoạch truy vấn từ Query Planner ở trên,
hãy viết câu lệnh SQL SELECT hoàn chỉnh cho SQLite.

Yêu cầu:
- Chỉ viết SELECT (không INSERT/UPDATE/DELETE)
- Đúng tên bảng và cột theo schema Chinook
- JOIN đúng foreign key relationships
- Thêm LIMIT 20 nếu không có LIMIT
- Chỉ trả về SQL thuần túy
""",
        expected_output="Một câu lệnh SQL SELECT hoàn chỉnh, chạy được trên SQLite",
        agent=generator,
        context=[task_plan],
    )

    task_execute = Task(
        description="""
Thực thi câu SQL vừa được tạo ra:
1. Dùng tool execute_sql để chạy câu SQL
2. Kiểm tra kết quả:
   - Nếu có lỗi SQL: mô tả lỗi chi tiết
   - Nếu kết quả rỗng (0 rows): ghi chú
   - Nếu thành công: xác nhận và trả về kết quả

Trả về: SQL đã chạy + kết quả đầy đủ từ database
""",
        expected_output="Kết quả thực thi SQL: câu SQL + dữ liệu trả về từ database",
        agent=executor,
        context=[task_generate],
    )

    task_interpret = Task(
        description=f"""
Câu hỏi gốc của người dùng: "{user_question}"

Dựa trên kết quả từ SQL Executor ở trên,
hãy viết câu trả lời bằng tiếng Việt dễ hiểu:
1. Trả lời trực tiếp câu hỏi
2. Trình bày kết quả chính (top items, con số quan trọng)
3. Nêu insight nếu có
4. Ngắn gọn, súc tích, thân thiện
""",
        expected_output="Câu trả lời bằng tiếng Việt dễ hiểu cho người dùng cuối",
        agent=interpreter,
        context=[task_execute],
    )

    return [task_plan, task_generate, task_execute, task_interpret]


# ── Hàm chạy pipeline ────────────────────────────────────────────────────────
def run_query(user_question: str) -> dict:
    """
    Chạy toàn bộ pipeline SQL Intelligence cho một câu hỏi.
    Trả về dict với: plan, sql, result, answer
    """
    print(f"\n{'='*60}")
    print(f"🔍 CÂU HỎI: {user_question}")
    print(f"{'='*60}\n")

    llm = get_llm()
    planner, generator, executor, interpreter = create_agents(llm)
    tasks = create_tasks(user_question, planner, generator, executor, interpreter)

    crew = Crew(
        agents=[planner, generator, executor, interpreter],
        tasks=tasks,
        process=Process.sequential,  # Chạy tuần tự: plan -> generate -> execute -> interpret
        verbose=True,
    )

    result = crew.kickoff()

    print(f"\n{'='*60}")
    print("✅ KẾT QUẢ CUỐI CÙNG:")
    print(f"{'='*60}")
    print(result.raw)

    return {
        "question": user_question,
        "answer": result.raw,
        "tasks_output": [t.output.raw if t.output else "" for t in tasks],
    }


# ── CLI Demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Các câu hỏi demo — đổi câu hỏi ở đây để test
    demo_questions = [
        "Top 3 nghệ sĩ có doanh thu cao nhất?",
        "Thể loại nhạc nào có nhiều bài hát nhất?",
        "Khách hàng từ quốc gia nào chi tiêu nhiều nhất?",
    ]

    question = demo_questions[0]

    # Hoặc nhập từ command line: python main.py "câu hỏi của bạn"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])

    run_query(question)
