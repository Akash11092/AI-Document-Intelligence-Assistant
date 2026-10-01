from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


output_file = "data/employee_handbook.pdf"

document = SimpleDocTemplate(
    output_file,
    pagesize=A4,
    rightMargin=50,
    leftMargin=50,
    topMargin=50,
    bottomMargin=50
)

styles = getSampleStyleSheet()

title_style = styles["Title"]
title_style.alignment = TA_CENTER

heading_style = styles["Heading2"]
body_style = styles["BodyText"]

content = []

content.append(
    Paragraph("EMPLOYEE HANDBOOK", title_style)
)
content.append(Spacer(1, 20))

sections = [
    (
        "1. Annual Leave Policy",
        """
        Full-time employees are entitled to 20 days of annual leave per calendar year.
        Leave requests should normally be submitted at least 3 working days before
        the requested leave date through the employee portal.
        Emergency leave may be requested without the normal advance notice.
        """
    ),
    (
        "2. Work From Home Policy",
        """
        Eligible employees may work from home up to 2 days per week.
        Work from home requires manager approval.
        Employees working remotely must remain available during normal working hours.
        Employees must maintain a stable internet connection while working remotely.
        """
    ),
    (
        "3. Working Hours",
        """
        Standard working hours are 9:00 AM to 6:00 PM, Monday to Friday.
        Employees receive a one-hour lunch break.
        Employees are expected to remain available during their scheduled working hours.
        """
    ),
    (
        "4. Notice Period",
        """
        The standard notice period for full-time employees is 30 days.
        Employees who wish to resign must submit a formal resignation request
        to their manager and the Human Resources department.
        """
    ),
    (
        "5. Remote Work Security",
        """
        Employees must use company-approved devices when accessing confidential
        company information remotely.
        Employees must not share company credentials with other individuals.
        Confidential company documents must not be stored on personal cloud storage services.
        """
    ),
    (
        "6. Employee Benefits",
        """
        Employees may be eligible for health insurance, paid leave,
        professional development programs, and other benefits according to
        their employment category and company policy.
        """
    ),
    (
        "7. Employee Support",
        """
        Employees can contact Human Resources for questions regarding leave,
        attendance, benefits, and workplace policies.
        Technical issues should be reported to the IT support team.
        """
    )
]

for heading, text in sections:
    content.append(Paragraph(heading, heading_style))
    content.append(Spacer(1, 8))
    content.append(Paragraph(text, body_style))
    content.append(Spacer(1, 15))

document.build(content)

print(f"PDF created successfully: {output_file}")