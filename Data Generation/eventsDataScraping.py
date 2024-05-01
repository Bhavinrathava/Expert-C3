from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import csv  # Import csv module

# Function to format date for URL
def format_date(date):
    return date.strftime('%Y-%m-%d')

# Setup Selenium WebDriver
options = Options()
options.headless = True
driver = webdriver.Chrome(options=options)

# Base URL and starting date
base_url = 'https://planitpurple.northwestern.edu/feed/124#search='
start_date = datetime.strptime('2024-05-03', '%Y-%m-%d')
days_to_scrape = 10

# Setup CSV file
with open('events_data_test1.csv', 'w', newline='', encoding='utf-8') as file:
    fieldnames = ['Date', 'Time', 'Title', 'Location', 'Link', 'Categories']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    # Scrape events for each day
    for i in range(days_to_scrape):
        date_to_scrape = start_date + timedelta(days=i)
        formatted_date = format_date(date_to_scrape)
        full_url = f'{base_url}{formatted_date}/0///'

        # Navigate to the page
        driver.get(full_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'ul.events')))

        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        events = soup.find_all('li')

        # Extract event details
        for event in events:
            title_link = event.find('a')
            title = title_link.get_text().strip() if title_link else 'N/A'
            link_suffix = title_link['href'] if title_link else ''
            link = f'https://planitpurple.northwestern.edu{link_suffix}'
            time = event.find('span', class_='event_time').get_text().strip() if event.find('span', class_='event_time') else 'N/A'
            location = event.find('span', class_='event_location').get_text().strip().lstrip('-').strip() if event.find('span', class_='event_location') else 'N/A'
            categories = ', '.join(tag.get_text().strip() for tag in event.find_all('span', class_='event_category')) if event.find('span', class_='event_category') else 'N/A'
            
            # Write event details to CSV
            writer.writerow({
                'Date': formatted_date,
                'Time': time,
                'Title': title,
                'Location': location,
                'Link': link,
                'Categories': categories
            })

# Close the browser
driver.quit()






"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# Set the base URL of the website
base_url = 'https://planitpurple.northwestern.edu/feed/124#search=2024-05-07/0///'

# Set the specific page URL
page_url = base_url + '/feed/124#search=2024-05-03/0///'

# Fetch the webpage
response = requests.get(page_url)
html_content = response.text

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

# Initialize a list to store event data
event_data = []

# Find all event date sections
event_days = soup.find_all('h3')
for day in event_days:
    date = day.get_text().strip()
    events_list = day.find_next_sibling('ul')
    events = events_list.find_all('li') if events_list else []
    for event in events:
        time = event.find('span', class_='event_time').get_text().strip() if event.find('span', class_='event_time') else 'N/A'
        event_details = event.find('span', class_='event_details')
        title_link = event_details.find('a') if event_details else None
        title = title_link.get_text().strip() if title_link else 'N/A'
        relative_link = title_link['href'] if title_link else 'N/A'
        link = os.path.join(base_url, relative_link)  # Construct the full URL
        location = event.find('span', class_='event_location').get_text().strip().lstrip('-').strip() if event.find('span', class_='event_location') else 'N/A'
        
        # Extract event categories
        category_tags = event.find_all('span', class_=lambda x: x and x.startswith('event_category'))
        categories = ', '.join([tag.get_text().strip() for tag in category_tags]) if category_tags else 'N/A'

        # Append the event information to the list
        event_data.append({'Date': date, 'Time': time, 'Title': title, 'Location': location, 'Link': link, 'Categories': categories})

# Create a DataFrame
df = pd.DataFrame(event_data)

# Define the file path
file_path = 'extractedEvents.csv'

# Check if file exists and append data accordingly
if os.path.exists(file_path):
    # File exists, append without header
    df.to_csv(file_path, mode='a', header=False, index=False)
else:
    # File does not exist, write with header
    df.to_csv(file_path, mode='w', header=True, index=False)

print(f"Data has been appended to '{file_path}' in the current working directory.")
"""