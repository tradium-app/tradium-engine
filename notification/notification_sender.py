# %%
import base64
from environs import Env
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Mail,
    Attachment,
    FileContent,
    FileName,
    FileType,
    Disposition,
)

env = Env()
env.read_env()
SENDGRID_API_KEY = env("SENDGRID_API_KEY")
NOTIFICATION_RECEIPNTS = env.list("NOTIFICATION_RECEIPNTS")


class Notification_Sender:
    def send(self):
        message = Mail(
            from_email="suraj.shrestha@live.com",
            to_emails=NOTIFICATION_RECEIPNTS,
            subject="TSLA Model Prediction from Tradium",
            html_content="<strong>predictions from Tradium</strong>",
        )

        with open("../charts/tsla-prediction.png", "rb") as f:
            data = f.read()
            f.close()
        encoded_file = base64.b64encode(data).decode()

        attachedFile = Attachment(
            FileContent(encoded_file),
            FileName("chart.png"),
            FileType("application/png"),
            Disposition("attachment"),
        )
        message.attachment = attachedFile

        try:
            sg = SendGridAPIClient(SENDGRID_API_KEY)
            response = sg.send(message)
            print(response.status_code, response.body, response.headers)
        except Exception as e:
            print(f"Printing exception: ", e)


if __name__ == "__main__":
    sender = Notification_Sender()
    sender.send()

