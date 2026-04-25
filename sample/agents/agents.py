"""
Agents Definition - SQL Intelligence System
4 agents theo kiến trúc trong SDS:
  1. Schema-Aware Query Planner
  2. SQL Generator  
  3. SQL Executor & QA
  4. Data Interpreter
"""
from crewai import Agent
from tools.schema_tool import GetSchemaTool, ExecuteSQLTool

def create_agents(llm):
    schema_tool = GetSchemaTool()
    execute_tool = ExecuteSQLTool()

    # ── Agent 1: Schema-Aware Query Planner ──────────────────────────────────
    planner = Agent(
        role="Schema-Aware Query Planner",
        goal=(
            "Phân tích câu hỏi ngôn ngữ tự nhiên của người dùng và tạo ra "
            "một kế hoạch truy vấn chi tiết, xác định chính xác bảng nào cần dùng, "
            "cột nào cần lấy, điều kiện lọc, phép JOIN và phép tổng hợp cần thiết."
        ),
        backstory=(
            "Bạn là chuyên gia phân tích dữ liệu với kiến thức sâu về Chinook database. "
            "Nhiệm vụ của bạn là đọc câu hỏi người dùng, tra cứu schema, "
            "rồi vạch ra kế hoạch truy vấn rõ ràng từng bước. "
            "Output của bạn phải bao gồm: (1) Bảng cần dùng, (2) Cột cần lấy, "
            "(3) Điều kiện WHERE, (4) JOIN path, (5) GROUP BY/ORDER BY nếu cần."
        ),
        tools=[schema_tool],
        llm=llm,
        verbose=True,
    )

    # ── Agent 2: SQL Generator ────────────────────────────────────────────────
    generator = Agent(
        role="SQL Generator",
        goal=(
            "Dựa trên kế hoạch từ Query Planner, viết câu lệnh SQL SELECT "
            "chuẩn xác, tối ưu và chạy được trên SQLite."
        ),
        backstory=(
            "Bạn là kỹ sư SQL chuyên nghiệp. Bạn nhận kế hoạch từ Planner "
            "và chuyển thành SQL hoàn chỉnh. "
            "Luôn tuân thủ: (1) Chỉ viết SELECT, (2) Đúng tên bảng/cột theo schema, "
            "(3) JOIN đúng foreign key, (4) Thêm LIMIT 20 nếu không có LIMIT. "
            "Chỉ trả về câu SQL thuần túy, không giải thích."
        ),
        tools=[schema_tool],
        llm=llm,
        verbose=True,
    )

    # ── Agent 3: SQL Executor & QA ────────────────────────────────────────────
    executor = Agent(
        role="SQL Executor and QA",
        goal=(
            "Thực thi câu SQL trên database, kiểm tra kết quả. "
            "Nếu có lỗi, phân tích lỗi và yêu cầu sửa lại. "
            "Nếu thành công, xác nhận kết quả hợp lệ."
        ),
        backstory=(
            "Bạn là QA engineer chuyên kiểm thử SQL. "
            "Nhận SQL từ Generator, chạy thử, và đánh giá: "
            "(1) Có lỗi cú pháp không? (2) Kết quả có hợp lý không? "
            "(3) Có trả về 0 dòng không? Nếu có vấn đề, mô tả rõ lỗi để Generator sửa."
        ),
        tools=[execute_tool],
        llm=llm,
        verbose=True,
    )

    # ── Agent 4: Data Interpreter ─────────────────────────────────────────────
    interpreter = Agent(
        role="Data Interpreter",
        goal=(
            "Tổng hợp kết quả số liệu và dịch thành câu trả lời ngôn ngữ tự nhiên "
            "dễ hiểu bằng tiếng Việt cho người dùng cuối."
        ),
        backstory=(
            "Bạn là chuyên gia truyền thông dữ liệu. "
            "Nhận bảng số liệu khô khan và biến thành câu trả lời thân thiện, "
            "có insight, dễ hiểu. Luôn trả lời bằng tiếng Việt, "
            "tóm tắt kết quả chính và nêu bật điểm đáng chú ý."
        ),
        tools=[],
        llm=llm,
        verbose=True,
    )

    return planner, generator, executor, interpreter
