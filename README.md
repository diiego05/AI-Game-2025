# GIẢI CÂU ĐỐ 8 Ô CHỮ (8 PUZZEL SOLVER) BẰNG ỨNG DỤNG NHIỀU THUẬT TOÁN TÌM KIẾM TRÍ TUỆ NHÂN TẠO VÀ TRỰC QUAN HÓA CÁC THUẬT TOÁN
## Lâm Văn Dỉ - 23110191
## 1.Mục tiêu
- Dự án game 8-puzzel được xây dựng nhằm mục đích triển khai các thuật toán trong 6 nhóm thuật toán:
  - Uninform Search (Tìm kiếm không có thông tin)
  - Informed Search / Heuristic Search (Tìm kiếm có thông tin)
  - Local Search (Tìm kiếm cục bộ)
  - Complex Environment Search (Tìm kiếm trong môi trường phức tạp)
  - Constraint-Based Search (Tìm kiếm trong môi trường có ràng buộc)
  - Reinforcement Learning (Học tăng cường)
- Cung cấp cho người dùng về hiệu suất giữa các thuật toán khác nhau trong 6 nhóm thuật toán

## 2.Nội dung
  ### 2.1 Các thuật toán tìm kiếm không có thông tin (Uninform Search)
  **Thành phần bài toán:**
  - Trạng thái: ma trận 3x3 gồm các số từ 0 đến 8 (0 là ô trống).
  - Hành động: di chuyển ô trống lên/xuống/trái/phải.
  - Chi phí đường đi: Tổng số bước di chuyển
  - Giải pháp (Solution): chuỗi các hành động dẫn đến trạng thái đích.
    
  **Các thuật toán trong nhóm**
  - **BFS (Breadth-First Search)** <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExN3l1ZTlyOGM2amF0ZG1rMnlocnhnY3FqZWY2MXcwNHhrcW83YW9leCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/GaOHKrpTiawcvoUjdZ/giphy.gif" alt="BFS" width="300"/>
  - **DFS (Depth-First Search)** <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExYTY1NnExam5wdTgzY2s0cTZ0c2NpaXBibHV0eWJ5ZDl3dG12ZHQ4eCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/BLLDvdAt5X7spkifaK/giphy.gif" alt="DFS" width="300"/>
  - **IDS (Iterative Deepening Search)** <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcXY2N2t3d3RsbjExcXppeThoMnA3a3loOHF4MXV0NXU0NXpnOHJpMyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/OgmVGLCc8j93VRUSJP/giphy.gif" alt="DFS" width="300"/>
  - **UCS (Uniform Cost Search)** <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExN2ZtZGhqcDN4b3pkaWxidnhld3g5dHd1anNtNW9xN2k0bnljdnA2biZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/OFBhqteP3EwAEelSFt/giphy.gif" alt="DFS" width="300"/>

   **Hiệu suất giữa các thuật toán** <img src="https://github.com/user-attachments/assets/3cfda135-790c-486c-9db8-ea5496adfcaf" alt="hieu suat trong nhom 1" width="300"/>
   **Nhận xét**
   - Có thể thấy BFS tốt nhất về tốc độ (2,857s) và tối ưu độ dài đường đi (23 steps)
   - DFS tuy có tốc độ nhanh (3,727s) nhưng về đường đi (29 steps) thì dài nhất trong 3 thuật toán còn lại 
   - 2 thuật toán còn lại là IDS và UCS đều có cùng độ dài đường đi với thuật toán BFS là 23 steps, tuy nhiên UCS (2,673s) lại có tốc độ nhanh hơn IDS (8,558s). Và hơn nữa là tuy IDS có số bước nhỏ nhưng tốc độ lại là lớn nhất
   - Kết luận trong nhóm này thuật toán chạy tốt nhất và nhanh nhất ưu tiên dùng nhiều hơn là BFS
     
  ### 2.2 Các thuật toán tìm kiếm có thông tin (Informed Search / Heuristic Search)
  **Thành phần bài toán:**
  - Trạng thái: ma trận 3x3 gồm các số từ 0 đến 8 (0 là ô trống).
  - Hành động: di chuyển ô trống lên/xuống/trái/phải.
  - Chi phí đường đi: Tổng số bước di chuyển
  - Hàm heuristic để dẫn đường tìm kiếm hiệu quả hơn.
  - Giải pháp (Solution): chuỗi các hành động dẫn đến trạng thái đích.
  **Các thuật toán trong nhóm**
  - **Greedy Search**
  - **A Star**
  - **IDA Star (Iterative Deepening A Star)**
  **Hiệu suất giữa các thuật toán**
  **Nhận xét**

  
  ### 2.3 Các thuật toán tìm kiếm có thông tin (Informed Search / Heuristic Search)
  **Thành phần bài toán:**
  - Trạng thái: ma trận 3x3 gồm các số từ 0 đến 8 (0 là ô trống).
  - Hành động: di chuyển ô trống lên/xuống/trái/phải.
  - Chi phí đường đi: Tổng số bước di chuyển
  - Hàm heuristic để dẫn đường tìm kiếm hiệu quả hơn.
  - Giải pháp (Solution): chuỗi các hành động dẫn đến trạng thái đích.
  **Các thuật toán trong nhóm**
  - **Greedy Search**
  - **A Star**
  - **IDA Star (Iterative Deepening A Star)**
  **Hiệu suất giữa các thuật toán**
  **Nhận xét**
    
