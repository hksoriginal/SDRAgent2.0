from Agents.EmailAgent.email_service import EmailService
from Agents.EmailAgent.email_extractor import EmailExtractor
from Agents.DataQueryAgent.load_data import DataLoader


class EmailAgent(EmailExtractor):
    def __init__(self):
        EmailExtractor.__init__(self)
        self.email_service = EmailService()
        self.data_loader = DataLoader()

    async def get_email_ids(self):
        leads_df = self.data_loader.load_data(
            file_path="Datafiles/filtered_leads.csv")
        emails_with_notes = self.get_lead_email_df(
            df=leads_df, email_col="Company", return_format="json")
        for email in emails_with_notes[:3]:
            email_service_res = await self.email_service.generate_email_content(
                context="You are an SDR agent tasked with reaching out to potential leads.",
                customer_info=str(email),
                product_info="Courses offered for all the specializations."
            )
            email_content = self.email_service.format_html(
                content=email_service_res.get("body"))

            await self.email_service.send_email(
                to="harshitkumar454@gmail.com",
                subject=email_service_res.get("subject"),
                html=email_content)

        return emails_with_notes
