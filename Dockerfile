# Dockerfile

# --- Stage 1: Base Image ---
# استفاده از یک ایمیج رسمی و سبک پایتون به عنوان پایه
FROM python:3.11-slim

# --- Set Environment Variables ---
# جلوگیری از ایجاد فایل‌های کش پایتون در کانتینر
ENV PYTHONDONTWRITEBYTECODE 1
# اطمینان از اینکه خروجی پایتون بدون بافر شدن مستقیماً به ترمینال ارسال می‌شود
ENV PYTHONUNBUFFERED 1

# --- Set Working Directory ---
# ایجاد یک پوشه کاری در داخل کانتینر برای نگهداری فایل‌های برنامه
WORKDIR /app

# --- Install Dependencies ---
# ابتدا فقط فایل نیازمندی‌ها کپی و نصب می‌شود. این یک بهینه‌سازی مهم برای سرعت بخشیدن
# به 빌دهای بعدی است. اگر این فایل تغییر نکند، داکر از کش استفاده می‌کند.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# --- Copy Application Code ---
# کپی کردن بقیه فایل‌های برنامه به داخل کانتینر
COPY . .

# --- Expose Port ---
# به داکر اعلام می‌کنیم که برنامه ما روی پورت 8000 کار خواهد کرد
EXPOSE 8000

# --- Command to Run Application ---
# دستوری که هنگام اجرای کانتینر برای راه‌اندازی سرور uvicorn اجرا می‌شود
# استفاده از --host 0.0.0.0 ضروری است تا برنامه از خارج کانتینر قابل دسترس باشد
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]