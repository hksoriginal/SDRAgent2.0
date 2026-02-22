import os
import asyncio
import logging
from typing import List, Union, Optional, Dict
import aiohttp
from dotenv import load_dotenv
from Mixins.llm_client import OpenRouterMixin
from Mixins.text_processor import TextProcessor
from Agents.EmailAgent.email_generation_prompt_template import EMAIL_GENERATION_PROMPT_TEMPLATE

load_dotenv()

logger = logging.getLogger(__name__)


llm_client = OpenRouterMixin()


class EmailService(TextProcessor):

    BASE_URL = "https://api.resend.com/emails"

    def __init__(self):
        self.api_key = os.getenv("RESEND_API_KEY")
        if not self.api_key:
            raise ValueError("Missing RESEND_API_KEY in environment variables")
        self.sender = os.getenv("SENDER_EMAIL", "harshit@resend.dev")
        logger.info("Resend API initialized successfully")

    def format_html(self, content: str) -> str:
        """
        Formats the email content into a clean, professional HTML structure
        with sender name and signature.
        """
        html_template = f"""
        <html>
            <head>
                <style>
                    body {{
                        font-family: 'Segoe UI', Arial, sans-serif;
                        background-color: #f4f6f8;
                        color: #333;
                        line-height: 1.6;
                        margin: 0;
                        padding: 40px;
                    }}
                    .container {{
                        background: #ffffff;
                        border-radius: 10px;
                        padding: 30px 40px;
                        max-width: 700px;
                        margin: 0 auto;
                        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
                    }}
                    .greeting {{
                        font-size: 1.05em;
                        margin-bottom: 15px;
                    }}
                    .content {{
                        margin-top: 10px;
                        font-size: 1em;
                    }}
                    .regards {{
                        margin-top: 30px;
                        font-size: 1em;
                    }}
                    .signature {{
                        font-weight: bold;
                        color: #0056b3;
                        margin-top: 5px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <p class="greeting">{content.split(',')[0].strip()},</p>

                    <div class="content">
                        {content.split(',')[1].strip()}
                    </div>

                    <div class="regards">
                        <p>Best regards,</p>
                        <p class="signature">Harshit Kumar<br>
                        Sales Development Representative<br>
                        Ema</p>
                    </div>
                </div>
            </body>
        </html>
        """
        return html_template.strip()

    async def generate_email_content(self, context: str, customer_info: str, product_info: str) -> Dict[str, str]:
        """
        Uses the LLM client to generate a professional, structured email body.
        """
        prompt = EMAIL_GENERATION_PROMPT_TEMPLATE.format(
            context=context,
            customer_info=customer_info,
            product_info=product_info
        )
        try:
            logger.info(
                "Generating email content via LLM with provided context and information.")
            result = await llm_client.chat_completion(messages=[{"role": "user", "content": prompt}])

            email_content = self.extract_and_validate(
                result, required_keys=["subject", "body"])
            return email_content
        except Exception as e:
            logger.exception("Failed to generate email content via LLM")
            raise RuntimeError(
                f"LLM email content generation failed: {e}") from e

    async def send_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        html: str,
        sender: Optional[str] = None,
        retries: int = 3,
        send: bool = True,
    ) -> dict:
        """
        Sends an email using the Resend API asynchronously.
        Retries up to `retries` times on failure.
        If send=False, just logs the email content instead of sending.
        """
        sender = sender or self.sender
        to_list = to if isinstance(to, list) else [to]

        payload = {
            "from": sender,
            "to": to_list,
            "subject": subject,
            "html": html,
        }

        if not send:
            logger.info(f"📧 Preview mode: Email not sent.\nPayload: {payload}")
            return {"status": "preview", "payload": payload}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.BASE_URL, json=payload, headers=headers) as response:
                        resp_data = await response.json()
                        if response.status == 200:
                            logger.info(
                                f"✅ Email sent successfully to {to_list}")
                            return resp_data
                        else:
                            logger.warning(
                                f"⚠️ Attempt {attempt}: Failed with {response.status} - {resp_data}")
            except Exception as e:
                logger.exception(f"Attempt {attempt} failed: {e}")

            await asyncio.sleep(2 ** attempt)  # exponential backoff

        raise RuntimeError(f"Email sending failed after {retries} attempts")
