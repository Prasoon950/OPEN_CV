
import cv2
import re
import pytesseract
 
image = cv2.imread(r'E:\DOWNLOAD\20200312_134443.jpg')


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#file_text = pytesseract.image_to_string(image, output_type="string", lang='eng')
#print(file_text)


emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", file_text)
numbers = re.findall(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]", file_text)


name = re.findall(r"[a-z0-9\.\-+_]+@", file_text)
name = [item.replace("@", "") for item in name]
name = [item.replace(".", " ") for item in name]

organisation  = re.findall(r"@[a-z0-9\.\-+_]+\.", file_text)
organisation = [item.replace("@", "") for item in organisation]
organisation = [item.replace(".", "") for item in organisation]

print("contact no-",numbers)
print("email id-", emails)
print("Name-", name)
print("organisation", organisation)