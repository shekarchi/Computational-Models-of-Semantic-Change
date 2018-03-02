import re
import sys
import time
import urllib.request
from bs4 import BeautifulSoup

#reload(sys)
#sys.setdefaultencoding('utf-8')

DATE_PATTERN = re.compile("\([a-z]*([0-9]{4})")

def main(word, start, end):
    url = "http://historicalthesaurus.arts.gla.ac.uk/category-selection/?qsearch={}".format(word)
    time.sleep(0.5)
    response = urllib.request.urlopen(url)
    page = response.read()
    soup = BeautifulSoup(page, 'html.parser')
    mainInner = soup.find('div', {"id": "mainInner"})
    counter = 0
    for p in mainInner.find_all('p', {"class": ['catEven', 'catOdd']}, recursive=False):
        text = p.find_all(text=True, recursive=False)
        match = DATE_PATTERN.search(text[-1])
        if not match:
            continue
        date = int(match.group(1))
        #print(date)
        if start <= date and date <= end:
            counter += 1
            return True
            pass
    return False


if __name__ == "__main__":
    #result = main("floor", 1900, 2000)
    least_file = sys.argv[1]
    most_file = sys.argv[2]

    least = open(least_file)
    least = least.readline().strip().split()[:100]
    least_counter = 0
    for w in least:
        result = main(w, 1925, 2000)
        print (w, result)
        if result == False:
            least_counter += 1
    print('The accuracy for least-changing words is: ', least_counter/len(least))
    most = open(most_file)
    most = most.readline().strip().split()[:100]
    most_counter = 0
    for w in most:
        result = main(w, 1925, 2000)
        print(w, result)
        if result == True:
            most_counter += 1
    print('The accuracy for most-changing words is: ', most_counter/len(most))






