import torch
import time

def infinite_loop():
    # GPU가 사용 가능한지 확인하고, 사용 가능하면 GPU를 사용합니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 무한 루프 시작
    while True:
        # 예시로 무작위 텐서 생성
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # 간단한 텐서 연산
        result = torch.mm(x, y)
        
        # 결과의 첫 번째 원소를 출력 (디버깅 목적으로)
        print(result[0, 0].item())
        
        # 잠시 대기 (과도한 자원 사용 방지)
        time.sleep(1)

if __name__ == "__main__":
    infinite_loop()
