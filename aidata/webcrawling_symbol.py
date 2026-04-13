from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import urllib.request
import time
import os
import builtins

# ✅ 1. 원본 img_crawler 함수 

def img_crawler():    
    query = input("검색어 : ")
    image_cnt = int(input("수집할 이미지 개수 : "))

    # 'H:\dzp\dog_cat\cat'
    save_dir = input("저장할 디렉토리 : ")
    
    driver = webdriver.Chrome()
    
    #os.makedirs(save_dir, exist_ok=True)  # 디렉토리 생성 (이미 존재하면 무시)
    #os.chdir(save_dir)  # 작업 디렉토리 변경
    
    os.makedirs(save_dir, exist_ok=True)
    # os.chdir(save_dir)  ← 이 줄 완전히 지워!


    URL = 'https://www.google.com/search?tbm=isch&q='
    driver.get(URL + query)  # 검색어를 포함한 URL로 이동

    # =======================================================
    # 무한 스크롤 처리
    # 스크롤 전 높이
    last_height = driver.execute_script("return window.scrollY")

    # 무한 스크롤
    while True:
        # 맨 아래로 스크롤을 내린다.
        driver.find_element(By.CSS_SELECTOR, "body").send_keys(Keys.END)
        
        # 스크롤 사이 페이지 로딩 시간
        time.sleep(1)
        
        # 스크롤 후 높이
        new_height = driver.execute_script("return window.scrollY")
        if new_height == last_height:
            break
        last_height = new_height
    # =======================================================
        

    # 이미지 정보 추출
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # 2024.08.23 duzin
    # g-img 태그의 mNsIhb 클래스 모두 조회
    image_info_list = driver.find_elements(By.CSS_SELECTOR, '.mNsIhb')
    
    image_and_name_list = []

    print('=== 이미지 수집 시작 === / ' + str(len(image_info_list)))
    
    downlaod_cnt = 0
    
    for i, image_info in enumerate(image_info_list):
        
        # 설정한 "수집할 이미지 수" 이상이면 빠지기~
        if i == image_cnt:
            break
        
        # 각 각 이미지 경로정보 가져오기
        save_image = image_info.find_element(By.CSS_SELECTOR, 'img').get_attribute('src')
        
        #image_path = os.path.join(query.replace(' ', '_') + '_' + str(downlaod_cnt) + '.jpg')
        image_path = os.path.join(save_dir, query.replace(' ', '_') + '_' + str(downlaod_cnt) + '.jpg')

        image_and_name_list.append((save_image, image_path))
        downlaod_cnt += 1

        print('    ※ ', i, '번째, ', save_image, ' 파일 다운로드가 완료되었습니다!')


    # Local 로 이미지 다운로드
    for i in range(len(image_and_name_list)):
        urllib.request.urlretrieve(image_and_name_list[i][0], image_and_name_list[i][1])

    print('=== 이미지 수집 종료 ===')
    driver.close()  # 브라우저 닫기

# ✅ 2. 자동화용 키워드 사전
# search_terms = {
#     "hand_wash": ["hand wash laundry symbol", "hand wash icon clothing"],
#     "do_not_wash": ["do not wash laundry symbol", "do not wash clothing icon"],
#     "machine_wash": ["washing machine laundry symbol", "machine wash clothing icon"],
#     "bleach_allowed": ["bleach laundry symbol", "triangle bleach symbol"],
#     "oxygen_bleach_only": ["non chlorine bleach symbol", "oxygen bleach laundry icon"],
#     "do_not_bleach": ["do not bleach laundry symbol", "no bleach clothing icon"],
#     "iron_allowed": ["iron laundry symbol", "iron allowed clothing icon"],
#     "low_temp_iron": ["low temperature iron symbol", "iron with one dot"],
#     "medium_temp_iron": ["medium temperature iron symbol", "iron with two dots"],
#     "high_temp_iron": ["high temperature iron symbol", "iron with three dots"],
#     "do_not_iron": ["do not iron laundry symbol", "no iron clothing icon"],
#     "dry_clean": ["dry cleaning laundry symbol", "dry clean icon on clothes"],
#     "do_not_dry_clean": ["do not dry clean symbol", "no dry clean laundry icon"],
#     "dry_clean_f": ["dry clean F symbol", "laundry symbol F"],
#     "dry_clean_p": ["dry clean P symbol", "laundry symbol P"],
#     "tumble_dry": ["tumble dry symbol", "dryer laundry icon"],
#     "tumble_dry_low": ["tumble dry low temperature symbol", "dryer symbol one dot"],
#     "tumble_dry_medium": ["tumble dry medium symbol", "dryer symbol two dots"],
#     "tumble_dry_high": ["tumble dry high temperature symbol", "dryer symbol three dots"],
#     "do_not_tumble_dry": ["do not tumble dry symbol", "no dryer laundry icon"]
# }

search_terms = {
    "wash_30": ["30 degree wash laundry symbol", "laundry symbol 30°C"],
    "wash_40": ["40 degree wash laundry symbol", "laundry symbol 40°C"],
    "wash_50": ["50 degree wash laundry symbol", "laundry symbol 50°C"],
    "wash_60": ["60 degree wash laundry symbol", "laundry symbol 60°C"],
    "wash_70": ["70 degree wash laundry symbol", "laundry symbol 70°C"],
    "wash_95": ["95 degree wash laundry symbol", "laundry symbol 95°C"],
    "wash_dot_1": ["1 dot wash symbol", "low temperature wash icon"],
    "wash_dot_2": ["2 dot wash symbol", "medium temperature wash icon"],
    "wash_dot_3": ["3 dot wash symbol", "high temperature wash icon"]
}



# ✅ 3. 자동 실행 루프 (오류 수정 완료)
def run_all_crawling():
    for class_name, keyword_list in search_terms.items():
        for idx, query in enumerate(keyword_list):
            save_dir = f"./images/{class_name}"
            image_cnt = "100"

            # ⚠️ 오류 수정된 부분
            inputs = iter([query, image_cnt, save_dir])
            builtins.input = lambda prompt="": next(inputs)

            print(f"\n🚀 [{class_name}] 키워드 {idx+1}: {query}")
            try:
                img_crawler()
                time.sleep(2)
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
                continue

# ✅ 실행
run_all_crawling()
