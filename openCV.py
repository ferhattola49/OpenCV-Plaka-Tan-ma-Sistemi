import cv2
import pytesseract
import imutils
import numpy as np


pytesseract.pytesseract.tesseract_cmd = r'C:/Tesseract-OCR\tesseract.exe'


# Görüntüyü yükleme
image = cv2.imread(r'C:/Users/Administrator/Desktop/OpenCV/plaka.jpg')

if image is None:
    print("Görüntü dosyası yüklenemedi. Lütfen dosya yolunu kontrol edin.")
    exit()

# Gri tonlamaya çevirme
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gürültü gidermek için bulanıklaştırma
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Kenarları tespit etmek için Canny kenar algılama
edged = cv2.Canny(blurred, 30, 150)

# Konturları bul
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# Konturları alanına göre sırala ve en büyükleri al
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

plaka_konturu = None
cropped = None  # Başlangıçta 'cropped' tanımlanıyor

for contour in contours:
    # Yaklaşık kontur hesaplama
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

    # Eğer konturun 4 köşesi varsa plaka olabilir
    if len(approx) == 4:
        plaka_konturu = approx
        break

# Eğer plaka konturu bulunduysa
if plaka_konturu is not None:
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [plaka_konturu], 0, 255, -1)
    x, y, w, h = cv2.boundingRect(plaka_konturu)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Yeşil dikdörtgen çiz
    cv2.putText(image, "Tespit Edilen Plaka", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Maske uygulandıktan sonra plakayı çıkar
    new_image = cv2.bitwise_and(image, image, mask=mask)

    # Plaka bölgesini kes
    (x, y) = np.where(mask == 255)
    (topX, topY, bottomX, bottomY) = (np.min(x), np.min(y), np.max(x), np.max(y))
    cropped = gray[topX:bottomX + 1, topY:bottomY + 1]

    # Plakayı OCR kullanarak tanıma
    plaka_metni = pytesseract.image_to_string(cropped, config='--psm 8')
    print(f"Tanınan plaka: {plaka_metni}")

    # Kesilen plaka ve sonuçları gösterme
    cv2.imshow("Plaka Bulundu", image)
    if cropped is not None:
        cv2.imshow("Kesilen Plaka", cropped)
else:
    print("Plaka bulunamadı.")


a = cv2.waitKey(0)
if a == 27:
    cv2.destroyAllWindows()
elif a == ord("s"):
    cv2.imwrite("Kesilen Plaka.jpg", cropped )
    cv2.imwrite("Tespit_edilen_plaka.jpg", image)



