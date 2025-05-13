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
  - **IDS (Iterative Deepening Search)** <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcXY2N2t3d3RsbjExcXppeThoMnA3a3loOHF4MXV0NXU0NXpnOHJpMyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/OgmVGLCc8j93VRUSJP/giphy.gif" alt="IDS" width="300"/>
  - **UCS (Uniform Cost Search)** <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExN2ZtZGhqcDN4b3pkaWxidnhld3g5dHd1anNtNW9xN2k0bnljdnA2biZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/OFBhqteP3EwAEelSFt/giphy.gif" alt="UCS" width="300"/>

   **Hiệu suất giữa các thuật toán** <img src="https://github.com/user-attachments/assets/3cfda135-790c-486c-9db8-ea5496adfcaf" alt="hieu suat trong nhom 1" width="300"/>
   
   **Nhận xét**
   - Có thể thấy BFS tốt nhất về tốc độ (2,857s) và tối ưu độ dài đường đi (23 steps)
   - DFS tuy có tốc độ nhanh (3,727s) nhưng về đường đi (29 steps) thì dài nhất trong 3 thuật toán còn lại 
   - 2 thuật toán còn lại là IDS và UCS đều có cùng độ dài đường đi với thuật toán BFS là 23 steps, tuy nhiên UCS (2,673s) lại có tốc độ nhanh hơn IDS (8,558s). Và hơn nữa là tuy IDS có số bước nhỏ nhưng tốc độ lại là lớn nhất
   - Kết luận trong nhóm này thuật toán chạy tốt nhất và nhanh nhất ưu tiên dùng nhiều hơn là BFS
     
  ### 2.2 Các thuật toán tìm kiếm có thông tin (Informed Search / Heuristic Search)
  - Các thuật toán này sử dụng hàm heuristic (trong code là manhattan_distance) để ước lượng khoảng cách còn lại đến đích. Có khả năng tìm đường hiệu quả hơn vì có định hướng
  **Thành phần bài toán:**
  - Trạng thái: ma trận 3x3 gồm các số từ 0 đến 8 (0 là ô trống).
  - Hành động: di chuyển ô trống lên/xuống/trái/phải.
  - Chi phí đường đi: Tổng số bước di chuyển
  - Hàm heuristic để dẫn đường tìm kiếm hiệu quả hơn.
  - Giải pháp (Solution): chuỗi các hành động dẫn đến trạng thái đích.
    
  **Các thuật toán trong nhóm**
  - **Greedy Search** <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExb3hoZWxjNjB2ZGRrcmN3dW84YjZpYjI3Z3lqYnhhMDYxOGc5MHFucyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/u03HBnGcDL8hN132hG/giphy.gif" alt="Greedy" width="300"/>
  - **A Star** <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExNG90YXNjbGxvb2xkaGJ0bm5kaHlxbnhueWU0NzNnb2Fib3NsNXV6aCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/uNKyhnsCUS4qH83H6x/giphy.gif" alt="A*" width="300"/>
  - **IDA Star (Iterative Deepening A Star)** <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExbHg5eDZycW92d2xtbHhpdnE5bTI5YnIzbmE5MjhrMW5kM3ozejJiZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WC2cRxvlxpcaWVYKB5/giphy.gif" alt="IDA*" width="300"/>
  **Hiệu suất giữa các thuật toán**
  **Nhận xét**

  
  ### 2.3 Các thuật toán tìm kiếm cục bộ (Local Search)
  - Không quan tâm đến đường đi, chỉ tập trung cải thiện trạng thái hiện tại.
  **Thành phần bài toán:**
  - Trạng thái:Một trạng thái có thể hợp lệ hoặc ngẫu nhiên.
  - Hàm đánh giá (Objective Function): khoảng cách Manhattan từ trạng thái hiện tại đến trạng thái đích.
  - Hàm lân cận: Trả về các trạng thái kế cận.
  - Giải pháp (Solution): chuỗi các hành động dẫn đến trạng thái đích.
    
  **Các thuật toán trong nhóm**
  - **Simple Hill Climbing** <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExcTRlZjZhbTBoZjBxYTVnNWczZ3VrcWh6ZHV6MTZ1dHZiZHR6NXNvayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/kbskoNgNfnAiOlhM1x/giphy.gif" alt="SHC" width="300"/>
  - **Steepest-Ascent Hill Climbing** <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExZmY5bTQ5ODVwZHQ2ankwdXdxNTFmbHE5c3JhYzJwcW8wamJhMXU3MiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/6liUIarF6oMdSR9E8U/giphy.gif" alt="SAHC" width="300"/> 
  - **Stochastic Hill Climbing**  <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExaG0xYXhhaHN4dmtxN3ByOHFyMG4wamE4dTNwY3FuM2pjOGoxb3FmdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/lA320cgDqNtoNWmSTK/giphy.gif" alt="StoHC" width="300"/>
  - **Simulated Annealing**  <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExeW51ZHI4MTZ3dTEyc3JiNHpndzh4ajFxdGllaDVya2V0MWl0bmg5OSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/jLP3xaJFDLFowPhngA/giphy.gif" alt="SA" width="300"/>
  - **Genetic Algorithm**  <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExdHBra2Zrcmx6NTkzZTBxeDdwbjQ1dTlmanh1NWxobmk4eHVsbWl2bCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/MeHYIIHbAA29CKfvBY/giphy.gif" alt="GA" width="300"/>
  - **Beam Search**  <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExZmMxY3hxaHE5amprMjF4eDE3MDZpbjZ5ZGxiZjlodmJmdnY3dTR3ZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/gQPy7HkN70BPuiyC1s/giphy.gif" alt="Beam" width="300"/>
  **Hiệu suất giữa các thuật toán**
  **Nhận xét**

  ### 2.4 Các thuật toán tìm kiếm trong môi trường phức tạp (Complex Environment Search)
  - Trạng thái không hoàn toàn quan sát được hoặc môi trường thay đổi. Trạng thái ban đầu có thể chứa None → cần sinh ra belief state.
  **Thành phần bài toán:**
  - Trạng thái không quan sát được hoàn toàn, hoặc thay đổi trong thời gian thực.
  - Cần xác định kế hoạch đảm bảo đạt đích trong mọi kịch bản.
  - Giải pháp (Solution): chuỗi các hành động dẫn đến trạng thái đích.
    
  **Các thuật toán trong nhóm**
  - **AND - OR Search** 
  - **Partially Observable (Sensorless Search)** <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzZibHpueTM1OWFubmg4OXptY3BlNXN5YnBkZWg4djRibnoxdTV4cCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/2LpgGVKWYrExi5XYko/giphy.gif" alt="Sensor" width="300"/>
  - **Unknown or Dynamic Environment (Không nhìn thấy hoàn toàn – tìm kiếm trong môi trường niềm tin)**
  **Hiệu suất giữa các thuật toán**
  **Nhận xét**
    
  ### 2.5 Các thuật toán tìm kiếm trong môi trường có ràng buộc (Constraint-Based Search - CSP)
  - Các thuật toán tìm kiếm trong môi trường có ràng buộc (Constraint-Based Search) được thiết kế để giải quyết bài toán thỏa mãn ràng buộc (Constraint Satisfaction Problem - CSP), nơi mà mục tiêu là tìm một hoặc nhiều giá trị cho các biến sao cho thỏa mãn một tập các ràng buộc đã cho
  **Thành phần bài toán:**
  - Trạng thái không quan sát được hoàn toàn, hoặc thay đổi trong thời gian thực.
  - Cần xác định kế hoạch đảm bảo đạt đích trong mọi kịch bản.
  - Giải pháp (Solution): chuỗi các hành động dẫn đến trạng thái đích.
    
  **Các thuật toán trong nhóm**
  - **Backtracking**
  - **AC3**
  **Hiệu suất giữa các thuật toán**
  **Nhận xét**

  ### 2.6 Học tăng cường (Reinforcement Learning)
  - Tìm chính sách hành động tối ưu (optimal policy) cho một agent (tác nhân) trong môi trường sao cho phần thưởng tích lũy là lớn nhất
  **Thành phần bài toán:**
  - Agent (tác tử) tương tác với môi trường để học hành động tối ưu qua thử và sai (trial and error)..
  - Phần thưởng (Reward) được cấp dựa trên hành động dẫn đến trạng thái mục tiêu.
  - Chính sách (Policy) dần được tối ưu hóa
  - Giải pháp (Solution): chuỗi các hành động dẫn đến trạng thái đích.
    
  **Các thuật toán trong nhóm**
  - **Q-Learning** 
  **Hiệu suất giữa các thuật toán**
  **Nhận xét**

