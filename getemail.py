
import imaplib
import email
import os
from email.header import decode_header
import re
# 邮箱配置
EMAIL =os.getenv('EMAIL')  # 你的邮箱地址
PASSWORD = os.getenv('PASSWORD')       # 你的邮箱密码
IMAP_SERVER = os.getenv('IMAP_SERVER') # IMAP 服务器地址
ATTACHMENT_DIR ='attachments'  # 附件下载目录
if not os.path.exists(ATTACHMENT_DIR):
    os.makedirs(ATTACHMENT_DIR)

# 创建附件保存目录
if not os.path.exists(ATTACHMENT_DIR):
    os.makedirs(ATTACHMENT_DIR)

# 连接到 IMAP 服务器
def connect_to_server():
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL, PASSWORD)
    return mail

# 清理文件名
def clean_filename(filename):
    # 解码文件名
    if filename.startswith('=?') and filename.endswith('?='):
        decoded = decode_header(filename)[0]
        if isinstance(decoded[0], bytes):
            filename = decoded[0].decode(decoded[1] or 'utf-8')
        else:
            filename = decoded[0]

    # 移除非法字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)  # 替换非法字符为下划线
    return filename

# 下载附件
def download_attachment(part, filename):
    filename = clean_filename(filename)  # 清理文件名
    filepath = os.path.join(ATTACHMENT_DIR, filename)
    with open(filepath, 'wb') as f:
        f.write(part.get_payload(decode=True))
    print(f"附件已下载: {filepath}")

# 获取邮件内容
def fetch_emails(mail, mailbox='INBOX', limit=5):
    # 选择邮箱文件夹
    mail.select(mailbox)

    # 搜索邮件
    status, messages = mail.search(None, 'ALL')
    if status != 'OK':
        print("无法获取邮件列表")
        return

    # 获取邮件 ID 列表
    message_ids = messages[0].split()
    for i, msg_id in enumerate(message_ids[-limit:]):  # 获取最新的 limit 封邮件
        # 获取邮件内容
        status, msg_data = mail.fetch(msg_id, '(RFC822)')
        if status != 'OK':
            print(f"无法获取邮件 {msg_id}")
            continue

        # 解析邮件
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or 'utf-8')
                from_ = msg.get("From")
                date = msg.get("Date")

                print(f"邮件 {i + 1}:")
                print(f"主题: {subject}")
                print(f"发件人: {from_}")
                print(f"日期: {date}")

                # 解析邮件正文和附件
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        try:
                            body = part.get_payload(decode=True).decode()
                        except:
                            body = None

                        if content_type == "text/plain" and "attachment" not in content_disposition:
                            print("正文:")
                            print(body)
                        elif "attachment" in content_disposition:
                            filename = part.get_filename()
                            if filename:
                                print(f"发现附件: {filename}")
                                download_attachment(part, filename)
                else:
                    body = msg.get_payload(decode=True).decode()
                    print("正文:")
                    print(body)

                print("-" * 50)

# 主函数
def main():
    mail = connect_to_server()
    fetch_emails(mail, limit=5)  # 获取最新的 5 封邮件
    mail.logout()

if __name__ == "__main__":
    main()