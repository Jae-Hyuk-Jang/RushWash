from bs4 import BeautifulSoup
from selenium import webdriver

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import urllib.request
import time
import os

def img_crawler():    
    query = input("검색어 : ")
    image_cnt = int(input("수집할 이미지 개수 : "))

    # 'H:\dzp\dog_cat\cat'
    save_dir = input("저장할 디렉토리 : ")
    
    driver = webdriver.Chrome()
    
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리 생성 (이미 존재하면 무시)
    os.chdir(save_dir)  # 작업 디렉토리 변경

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
        
        image_path = os.path.join(query.replace(' ', '_') + '_' + str(downlaod_cnt) + '.jpg')
        image_and_name_list.append((save_image, image_path))
        downlaod_cnt += 1

        print('    ※ ', i, '번째, ', save_image, ' 파일 다운로드가 완료되었습니다!')


    # Local 로 이미지 다운로드
    for i in range(len(image_and_name_list)):
        urllib.request.urlretrieve(image_and_name_list[i][0], image_and_name_list[i][1])

    print('=== 이미지 수집 종료 ===')
    driver.close()  # 브라우저 닫기
    

img_crawler()