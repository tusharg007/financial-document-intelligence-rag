"""
Pre-loaded sample financial data for demo mode.

Contains realistic SEC filing excerpts from major companies across multiple
filing types and time periods. This enables the Streamlit demo to work
without any API keys or external data downloads.
"""
from typing import List, Dict, Any

# Each document chunk includes: content, company, filing_type, section,
# filing_date, fiscal_year, fiscal_quarter


SAMPLE_DOCUMENTS: List[Dict[str, Any]] = [
    # =========================================================================
    # TESLA 10-K FILINGS
    # =========================================================================
    {
        "content": (
            "We face significant risks related to our ability to ramp production "
            "of our vehicles and energy storage products. Our production processes "
            "are highly complex and involve significant risks and uncertainties, "
            "including with respect to our ability to maintain quality standards "
            "as we increase production volume. We have in the past experienced, "
            "and may in the future experience, parsing and qualification issues "
            "with components, production line shutdowns or slowdowns due to "
            "supply chain interruptions, equipment malfunctions, or quality "
            "control issues. Any failure to maintain quality or efficiency as we "
            "scale could materially harm our brand, business, and financial results."
        ),
        "company": "Tesla Inc.",
        "ticker": "TSLA",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-01-29",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0001318605"
    },
    {
        "content": (
            "We are highly dependent on the services of Elon Musk, our Chief "
            "Executive Officer, product architect, and Chairman of our Board of "
            "Directors. Although Mr. Musk spends significant time with Tesla and "
            "is highly active in our management, he also allocates time to his "
            "other ventures, including SpaceX, Neuralink, The Boring Company, "
            "xAI, and X (formerly Twitter). We are dependent on the continued "
            "services and performance of our key personnel. The loss of any key "
            "personnel could disrupt our operations, slow development of our "
            "products, and adversely affect our business."
        ),
        "company": "Tesla Inc.",
        "ticker": "TSLA",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-01-29",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0001318605"
    },
    {
        "content": (
            "Total revenue for the fiscal year ended December 31, 2023 was "
            "$96.77 billion, an increase of 19% compared to $81.46 billion for "
            "the fiscal year ended December 31, 2022. Automotive revenue was "
            "$82.42 billion for 2023 compared to $71.46 billion for 2022, an "
            "increase of 15%. Energy generation and storage revenue was $6.04 "
            "billion for 2023, representing a 54% increase year-over-year. "
            "Services and other revenue was $8.32 billion, an increase of 37%. "
            "We delivered approximately 1.81 million vehicles during 2023, "
            "compared to approximately 1.31 million vehicles during 2022."
        ),
        "company": "Tesla Inc.",
        "ticker": "TSLA",
        "filing_type": "10-K",
        "section": "Management Discussion and Analysis",
        "filing_date": "2024-01-29",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0001318605"
    },
    {
        "content": (
            "Our supply chain is global and complex, and we face risks related "
            "to the availability and cost of raw materials, including lithium, "
            "nickel, cobalt, manganese, and other materials used in our battery "
            "cells. The prices of these materials have been volatile and may "
            "continue to fluctuate. We are working to reduce our dependence on "
            "certain critical minerals through innovations in battery chemistry, "
            "including our development of lithium iron phosphate (LFP) cells and "
            "our proprietary 4680 battery cell technology, which aims to reduce "
            "cost per kWh while improving energy density and manufacturing efficiency."
        ),
        "company": "Tesla Inc.",
        "ticker": "TSLA",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-01-29",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0001318605"
    },
    {
        "content": (
            "Gross profit for 2023 was $17.66 billion, a decrease from $20.85 "
            "billion in 2022. Automotive gross margin decreased to 18.2% in 2023 "
            "from 25.6% in 2022, primarily due to price reductions across our "
            "vehicle lineup implemented to stimulate demand, increased raw "
            "material costs, and production ramp-up costs for the Cybertruck and "
            "other new initiatives. We expect continued pressure on margins as "
            "we navigate a competitive pricing environment while investing in "
            "next-generation vehicle platforms."
        ),
        "company": "Tesla Inc.",
        "ticker": "TSLA",
        "filing_type": "10-K",
        "section": "Management Discussion and Analysis",
        "filing_date": "2024-01-29",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0001318605"
    },
    # Tesla Q3 2024 10-Q
    {
        "content": (
            "For the three months ended September 30, 2024, total revenue was "
            "$25.18 billion compared to $23.35 billion for Q3 2023, an increase "
            "of 8%. Automotive revenue was $20.02 billion. We delivered "
            "approximately 462,890 vehicles during Q3 2024, an increase of 6% "
            "year-over-year. Automotive gross margin was 17.1% in Q3 2024 "
            "compared to 16.3% in Q3 2023, reflecting improved manufacturing "
            "efficiency and cost reduction efforts partially offset by continued "
            "competitive pricing dynamics."
        ),
        "company": "Tesla Inc.",
        "ticker": "TSLA",
        "filing_type": "10-Q",
        "section": "Management Discussion and Analysis",
        "filing_date": "2024-10-23",
        "fiscal_year": "2024",
        "fiscal_quarter": "Q3",
        "cik": "0001318605"
    },
    {
        "content": (
            "Our Full Self-Driving (FSD) technology remains a significant area "
            "of investment and development. As of Q3 2024, FSD (Supervised) has "
            "been deployed to approximately 500,000 vehicles in North America. "
            "While we continue to make progress in autonomous driving capability, "
            "regulatory uncertainty remains a key risk factor. Different "
            "jurisdictions have varying approaches to autonomous vehicle "
            "regulation, and changes in laws or regulations could impact our "
            "ability to deploy or monetize autonomous driving technology."
        ),
        "company": "Tesla Inc.",
        "ticker": "TSLA",
        "filing_type": "10-Q",
        "section": "Risk Factors",
        "filing_date": "2024-10-23",
        "fiscal_year": "2024",
        "fiscal_quarter": "Q3",
        "cik": "0001318605"
    },
    # =========================================================================
    # APPLE 10-K FILINGS 
    # =========================================================================
    {
        "content": (
            "The Company's business, reputation and financial condition are "
            "subject to risks associated with global supply chain disruptions. "
            "The Company's products and services rely on components and "
            "manufacturing from partners primarily located in China, India, "
            "Japan, South Korea, Taiwan, and the United States. Supply chain "
            "disruptions, whether due to geopolitical tensions, natural "
            "disasters, pandemics, trade restrictions, or logistics challenges, "
            "could adversely affect the Company's ability to meet customer "
            "demand and negatively impact revenue and profitability."
        ),
        "company": "Apple Inc.",
        "ticker": "AAPL",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-11-01",
        "fiscal_year": "2024",
        "fiscal_quarter": "FY",
        "cik": "0000320193"
    },
    {
        "content": (
            "Total net revenue for fiscal year 2024 was $391.0 billion, "
            "compared to $383.3 billion in fiscal year 2023, an increase of "
            "2%. Products revenue was $298.1 billion in 2024! versus $298.1 "
            "billion in 2023, remaining essentially flat. Services revenue "
            "grew 13% to $92.9 billion from $85.2 billion. iPhone revenue "
            "was $201.2 billion, Mac revenue was $29.9 billion, iPad revenue "
            "was $26.7 billion, and Wearables, Home, and Accessories revenue "
            "was $37.0 billion. Greater China revenue declined 6% to $67.0 "
            "billion, reflecting ongoing macroeconomic challenges in the region."
        ),
        "company": "Apple Inc.",
        "ticker": "AAPL",
        "filing_type": "10-K",
        "section": "Management Discussion and Analysis",
        "filing_date": "2024-11-01",
        "fiscal_year": "2024",
        "fiscal_quarter": "FY",
        "cik": "0000320193"
    },
    {
        "content": (
            "The Company is subject to complex and evolving laws and regulations "
            "regarding privacy, data protection, and data security in the U.S. "
            "and internationally. The European Union's General Data Protection "
            "Regulation (GDPR), the Digital Markets Act (DMA), and similar "
            "legislation in other jurisdictions impose stringent requirements "
            "on how the Company collects, processes, stores, and shares personal "
            "data. Compliance with these regulations requires significant "
            "resources and may constrain the Company's business operations. "
            "The DMA in particular has required changes to how we operate the "
            "App Store and other services in Europe."
        ),
        "company": "Apple Inc.",
        "ticker": "AAPL",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-11-01",
        "fiscal_year": "2024",
        "fiscal_quarter": "FY",
        "cik": "0000320193"
    },
    {
        "content": (
            "The Company continues to invest heavily in research and development "
            "to drive innovation across its product and service offerings. R&D "
            "expense was $31.4 billion in fiscal 2024 compared to $29.9 billion "
            "in fiscal 2023. Key areas of investment include Apple Intelligence "
            "(the Company's AI and machine learning platform), Apple Vision Pro "
            "and spatial computing, custom silicon development (M-series and "
            "A-series chips), health technologies, and services innovation. "
            "The Company believes that focused investment in R&D is critical "
            "to its future growth and competitive position."
        ),
        "company": "Apple Inc.",
        "ticker": "AAPL",
        "filing_type": "10-K",
        "section": "Management Discussion and Analysis",
        "filing_date": "2024-11-01",
        "fiscal_year": "2024",
        "fiscal_quarter": "FY",
        "cik": "0000320193"
    },
    # =========================================================================
    # FORD 10-K FILINGS
    # =========================================================================
    {
        "content": (
            "Our transition to electric vehicles presents significant "
            "operational and financial risks. We have committed to investing "
            "over $50 billion in electric vehicles through 2026. The EV market, "
            "however, has experienced slower-than-expected adoption rates, "
            "leading us to adjust our production plans and capital allocation. "
            "Ford Model e, our EV segment, reported an EBIT loss of $4.7 "
            "billion in 2023, and we expect continued losses in 2024 as we "
            "work to reduce costs and improve the economics of our EV lineup "
            "including the Mustang Mach-E and F-150 Lightning."
        ),
        "company": "Ford Motor Company",
        "ticker": "F",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-02-06",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0000037996"
    },
    {
        "content": (
            "Total company revenue for 2023 was $176.2 billion compared to "
            "$158.1 billion in 2022, an increase of 11%. Ford Blue (ICE vehicles) "
            "generated revenue of $113.8 billion with an EBIT of $7.5 billion. "
            "Ford Pro (commercial vehicles and services) generated revenue of "
            "$61.8 billion with an EBIT of $7.2 billion. Ford Model e (electric "
            "vehicles) generated revenue of $6.4 billion with an EBIT loss of "
            "$4.7 billion. We sold approximately 4.4 million vehicles worldwide "
            "in 2023, with strong performance in our truck and SUV segments."
        ),
        "company": "Ford Motor Company",
        "ticker": "F",
        "filing_type": "10-K",
        "section": "Management Discussion and Analysis",
        "filing_date": "2024-02-06",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0000037996"
    },
    {
        "content": (
            "We face significant supply chain risks including semiconductor "
            "shortages, raw material cost volatility, and logistics disruptions. "
            "The global semiconductor supply situation has improved from the "
            "acute shortages experienced in 2021-2022, but localized supply "
            "constraints persist and may recur. Our supply chain involves "
            "approximately 1,400 Tier 1 suppliers across multiple continents, "
            "and disruption at any point can impact our production schedules. "
            "We are actively working to diversify our supply base and build "
            "strategic buffer inventory for critical components."
        ),
        "company": "Ford Motor Company",
        "ticker": "F",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-02-06",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0000037996"
    },
    {
        "content": (
            "Our warranty costs remain a significant concern. Warranty expense "
            "and related costs were $5.2 billion in 2023 compared to $4.8 "
            "billion in 2022. We are implementing Ford+ quality initiatives "
            "aimed at reducing warranty costs by 25% by 2025 through improved "
            "design, manufacturing processes, and supplier quality standards. "
            "Quality issues can damage our reputation, increase costs, and "
            "result in regulatory actions or litigation."
        ),
        "company": "Ford Motor Company",
        "ticker": "F",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-02-06",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0000037996"
    },
    # Ford Q3 2024 10-Q
    {
        "content": (
            "Ford Model e continues to face headwinds with an EBIT loss of "
            "$1.2 billion in Q3 2024. We have strategically delayed certain "
            "next-generation EV programs to focus on cost reduction and "
            "profitability. The three-row electric SUV has been pushed to 2027, "
            "and we are pivoting investments toward hybrid vehicles where we "
            "see stronger near-term customer demand. Ford Pro continues to be "
            "our strongest profit driver with Q3 2024 EBIT of $1.8 billion."
        ),
        "company": "Ford Motor Company",
        "ticker": "F",
        "filing_type": "10-Q",
        "section": "Management Discussion and Analysis",
        "filing_date": "2024-10-28",
        "fiscal_year": "2024",
        "fiscal_quarter": "Q3",
        "cik": "0000037996"
    },
    # =========================================================================
    # MICROSOFT 10-K FILINGS
    # =========================================================================
    {
        "content": (
            "Revenue was $245.1 billion for fiscal year 2024, an increase of "
            "16% compared to $211.9 billion for fiscal year 2023. Revenue from "
            "Intelligent Cloud was $105.4 billion, an increase of 22%, driven "
            "primarily by Azure and other cloud services revenue growth of 30%. "
            "Productivity and Business Processes revenue was $79.4 billion, "
            "an increase of 12%, primarily driven by Office 365 Commercial and "
            "LinkedIn. More Personal Computing revenue was $60.3 billion, an "
            "increase of 9%, driven by Xbox content and services and Windows."
        ),
        "company": "Microsoft Corporation",
        "ticker": "MSFT",
        "filing_type": "10-K",
        "section": "Management Discussion and Analysis",
        "filing_date": "2024-07-30",
        "fiscal_year": "2024",
        "fiscal_quarter": "FY",
        "cik": "0000789019"
    },
    {
        "content": (
            "The AI landscape is evolving rapidly, and we face significant "
            "competitive risks as well as risks related to the responsible "
            "development and deployment of AI technologies. We have invested "
            "billions of dollars in our partnership with OpenAI and in building "
            "our own AI infrastructure. Competitive dynamics in AI are intense, "
            "with significant investments being made by Google, Amazon, Meta, "
            "Apple, and numerous startups. If our AI products and services do "
            "not achieve widespread adoption, or if competitors develop superior "
            "AI capabilities, our substantial investments may not generate "
            "expected returns."
        ),
        "company": "Microsoft Corporation",
        "ticker": "MSFT",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-07-30",
        "fiscal_year": "2024",
        "fiscal_quarter": "FY",
        "cik": "0000789019"
    },
    {
        "content": (
            "Capital expenditures were $44.5 billion for fiscal year 2024, "
            "a significant increase from $28.1 billion in fiscal 2023, primarily "
            "driven by investments in cloud and AI infrastructure, including "
            "datacenters and specialized hardware (GPUs). We expect capital "
            "expenditures to continue increasing in fiscal 2025 as we expand "
            "our AI and cloud capacity to meet growing customer demand. The "
            "return on these capital investments depends on continued growth "
            "in demand for our cloud and AI services."
        ),
        "company": "Microsoft Corporation",
        "ticker": "MSFT",
        "filing_type": "10-K",
        "section": "Management Discussion and Analysis",
        "filing_date": "2024-07-30",
        "fiscal_year": "2024",
        "fiscal_quarter": "FY",
        "cik": "0000789019"
    },
    {
        "content": (
            "We are subject to cybersecurity threats and incidents that could "
            "adversely affect our business. In January 2024, we disclosed that "
            "a nation-state actor (Midnight Blizzard) had accessed certain "
            "executive email accounts. This incident highlighted the evolving "
            "nature of cybersecurity threats and the importance of our Secure "
            "Future Initiative (SFI), which we launched to enhance our security "
            "posture across all products and services. Cybersecurity remains "
            "one of our top enterprise risk categories."
        ),
        "company": "Microsoft Corporation",
        "ticker": "MSFT",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-07-30",
        "fiscal_year": "2024",
        "fiscal_quarter": "FY",
        "cik": "0000789019"
    },
    # Microsoft Q1 FY2025 10-Q
    {
        "content": (
            "Revenue was $65.6 billion for Q1 FY2025, an increase of 16% "
            "year-over-year. Azure and other cloud services revenue grew 34%, "
            "driven by strong demand for AI services. Microsoft 365 Copilot "
            "adoption continued to accelerate, with over 1 million paid users "
            "by the end of the quarter. AI-related revenue run rate exceeded "
            "$10 billion annualized, demonstrating strong monetization of our "
            "AI investments across the product portfolio."
        ),
        "company": "Microsoft Corporation",
        "ticker": "MSFT",
        "filing_type": "10-Q",
        "section": "Management Discussion and Analysis",
        "filing_date": "2024-10-30",
        "fiscal_year": "2025",
        "fiscal_quarter": "Q1",
        "cik": "0000789019"
    },
    # =========================================================================
    # NVIDIA 10-K FILINGS
    # =========================================================================
    {
        "content": (
            "Revenue for fiscal year 2024 was $60.9 billion, an increase of "
            "126% from $27.0 billion in fiscal 2023. Data Center revenue was "
            "$47.5 billion, an increase of 217%, driven by strong demand for "
            "our AI training and inference platforms including H100 and A100 "
            "GPUs. The adoption of generative AI and large language models by "
            "cloud service providers, enterprises, and consumer internet "
            "companies has driven unprecedented demand for our data center "
            "products."
        ),
        "company": "NVIDIA Corporation",
        "ticker": "NVDA",
        "filing_type": "10-K",
        "section": "Management Discussion and Analysis",
        "filing_date": "2024-02-21",
        "fiscal_year": "2024",
        "fiscal_quarter": "FY",
        "cik": "0001045810"
    },
    {
        "content": (
            "We face concentration risk as a significant portion of our data "
            "center revenue is derived from a limited number of large customers, "
            "including major cloud service providers. Additionally, U.S. "
            "government export controls imposed on advanced semiconductor "
            "technologies, including restrictions on shipments to China and "
            "certain other countries, have impacted and may continue to impact "
            "our revenue. We estimate these export restrictions reduced our "
            "addressable market by several billion dollars in fiscal 2024."
        ),
        "company": "NVIDIA Corporation",
        "ticker": "NVDA",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-02-21",
        "fiscal_year": "2024",
        "fiscal_quarter": "FY",
        "cik": "0001045810"
    },
    {
        "content": (
            "Supply chain constraints remain a significant risk factor. Our "
            "products are manufactured by third-party foundries, primarily "
            "Taiwan Semiconductor Manufacturing Company (TSMC). Our reliance "
            "on a single foundry for our most advanced products creates "
            "concentration risk. Geopolitical tensions regarding Taiwan, "
            "natural disasters, or capacity constraints at TSMC could "
            "significantly impact our ability to meet customer demand. We have "
            "entered into long-term supply agreements and prepaid supply "
            "arrangements totaling approximately $9.3 billion to secure "
            "manufacturing capacity."
        ),
        "company": "NVIDIA Corporation",
        "ticker": "NVDA",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-02-21",
        "fiscal_year": "2024",
        "fiscal_quarter": "FY",
        "cik": "0001045810"
    },
    # =========================================================================
    # JPMORGAN CHASE 10-K FILINGS
    # =========================================================================
    {
        "content": (
            "Net revenue for 2023 was $162.4 billion, an increase of 23% "
            "from $132.3 billion in 2022. Net income was $49.6 billion, or "
            "$16.23 per diluted share, compared with $37.7 billion, or $12.09 "
            "per diluted share in 2022. The increase in revenue was primarily "
            "driven by higher net interest income due to the higher rate "
            "environment, partially offset by lower noninterest revenue. "
            "Return on common equity was 17%, above our through-the-cycle "
            "target of 17%."
        ),
        "company": "JPMorgan Chase & Co.",
        "ticker": "JPM",
        "filing_type": "10-K",
        "section": "Management Discussion and Analysis",
        "filing_date": "2024-02-16",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0000019617"
    },
    {
        "content": (
            "The global economic environment remains uncertain, with risks "
            "including persistent inflation, potential recession, geopolitical "
            "conflicts (including the wars in Ukraine and the Middle East), "
            "commercial real estate stress, and evolving regulatory requirements. "
            "Credit quality remained strong in 2023 but we expect normalization "
            "of credit costs. Our provision for credit losses was $9.8 billion "
            "in 2023. We maintain a fortress balance sheet with CET1 capital "
            "ratio of 15.0% and total loss-absorbing capacity of approximately "
            "$505 billion."
        ),
        "company": "JPMorgan Chase & Co.",
        "ticker": "JPM",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-02-16",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0000019617"
    },
    {
        "content": (
            "We believe AI will fundamentally transform financial services. "
            "JPMorgan Chase has over 2,000 AI and machine learning use cases "
            "in production, spanning fraud detection, risk management, customer "
            "service, trading strategies, and compliance. We employ over 2,000 "
            "data scientists and AI/ML engineers. In 2023, we launched IndexGPT "
            "and LLM Suite for internal use, enabling employees to leverage "
            "large language models securely. We continue to invest significantly "
            "in our technology infrastructure, with a technology budget of "
            "approximately $15.8 billion in 2023."
        ),
        "company": "JPMorgan Chase & Co.",
        "ticker": "JPM",
        "filing_type": "10-K",
        "section": "Management Discussion and Analysis",
        "filing_date": "2024-02-16",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0000019617"
    },
    # =========================================================================
    # AMAZON 10-K FILINGS
    # =========================================================================
    {
        "content": (
            "Net sales increased 12% to $574.8 billion in 2023, compared with "
            "$514.0 billion in 2022. AWS segment sales were $90.8 billion, "
            "an increase of 13%, reflecting continued customer adoption of "
            "cloud services including new generative AI capabilities through "
            "Amazon Bedrock. North America segment sales were $352.8 billion, "
            "and International segment sales were $131.2 billion. Operating "
            "income was $36.9 billion compared to $12.2 billion in 2022, a "
            "significant improvement driven by operational efficiency programs "
            "and revenue growth."
        ),
        "company": "Amazon.com Inc.",
        "ticker": "AMZN",
        "filing_type": "10-K",
        "section": "Management Discussion and Analysis",
        "filing_date": "2024-02-01",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0001018724"
    },
    {
        "content": (
            "We face intense competition in each of our business segments. "
            "In our e-commerce businesses, we compete with a large number of "
            "online and brick-and-mortar retailers. AWS faces intense "
            "competition from Microsoft Azure, Google Cloud Platform, Oracle "
            "Cloud, and others. The rapid growth of generative AI has intensified "
            "cloud competition. Supply chain challenges, including transportation "
            "costs, labor availability, and logistics network optimization, "
            "continue to be important factors in our ability to maintain "
            "customer satisfaction and manage operating costs."
        ),
        "company": "Amazon.com Inc.",
        "ticker": "AMZN",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-02-01",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0001018724"
    },
    # =========================================================================
    # GOOGLE (ALPHABET) 10-K FILINGS
    # =========================================================================
    {
        "content": (
            "Alphabet's total revenue was $307.4 billion for fiscal year 2023, "
            "an increase of 9% from $282.8 billion in 2022. Google Search & "
            "other advertising revenue was $175.0 billion, an increase of 9%. "
            "YouTube ads revenue was $31.5 billion, an increase of 7%. Google "
            "Cloud revenue grew 26% to $33.1 billion and achieved operating "
            "profitability for the first full year. Other Bets revenue was $1.5 "
            "billion with an operating loss of $4.8 billion. Total headcount "
            "decreased by approximately 1,400 in 2023 following restructuring."
        ),
        "company": "Alphabet Inc.",
        "ticker": "GOOGL",
        "filing_type": "10-K",
        "section": "Management Discussion and Analysis",
        "filing_date": "2024-01-30",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0001652044"
    },
    {
        "content": (
            "The rapid development and deployment of AI technologies presents "
            "both opportunities and risks. We are embedding AI across our "
            "products and services through investments in Gemini, our most "
            "capable AI model, and through Google Cloud's Vertex AI platform. "
            "However, AI technologies create risks including potential for "
            "generating harmful or biased outputs, intellectual property "
            "concerns, competitive displacement, regulatory actions, and "
            "significant capital requirements. Our total AI-related capital "
            "expenditure was approximately $32 billion in 2023."
        ),
        "company": "Alphabet Inc.",
        "ticker": "GOOGL",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-01-30",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0001652044"
    },
    # =========================================================================
    # 8-K FILINGS (Material Events)
    # =========================================================================
    {
        "content": (
            "On October 23, 2024, Tesla, Inc. reported third-quarter 2024 "
            "financial results. Total revenue was $25.2 billion, exceeding "
            "Wall Street expectations. The company reported GAAP net income "
            "of $2.2 billion, or $0.62 per share. Non-GAAP earnings per share "
            "were $0.72, beating consensus estimates of $0.58. The company "
            "generated $2.7 billion in free cash flow during the quarter. "
            "Tesla reiterated its expectation for slight vehicle delivery "
            "growth in 2024 compared to 2023."
        ),
        "company": "Tesla Inc.",
        "ticker": "TSLA",
        "filing_type": "8-K",
        "section": "Earnings Release",
        "filing_date": "2024-10-23",
        "fiscal_year": "2024",
        "fiscal_quarter": "Q3",
        "cik": "0001318605"
    },
    {
        "content": (
            "On January 25, 2024, Microsoft Corporation reported fiscal Q2 "
            "2024 results. Revenue was $62.0 billion, an increase of 18% "
            "year-over-year. Azure and other cloud services revenue grew 30%. "
            "Microsoft Cloud revenue was $33.7 billion, up 24%. The company "
            "highlighted growing momentum in AI services, with Copilot "
            "adoption expanding across enterprise customers. Operating income "
            "was $27.0 billion, an increase of 33%."
        ),
        "company": "Microsoft Corporation",
        "ticker": "MSFT",
        "filing_type": "8-K",
        "section": "Earnings Release",
        "filing_date": "2024-01-25",
        "fiscal_year": "2024",
        "fiscal_quarter": "Q2",
        "cik": "0000789019"
    },
    {
        "content": (
            "Ford Motor Company announced on February 6, 2024 that it is "
            "restructuring its electric vehicle strategy, reallocating "
            "approximately $12 billion in planned EV investments. The company "
            "will delay certain next-generation EV programs and shift capital "
            "toward hybrid vehicle development, where it sees stronger near-"
            "term demand. Ford cited slower-than-expected EV adoption rates "
            "and the need to improve EV profitability as key factors in the "
            "strategic adjustment."
        ),
        "company": "Ford Motor Company",
        "ticker": "F",
        "filing_type": "8-K",
        "section": "Material Event",
        "filing_date": "2024-02-06",
        "fiscal_year": "2024",
        "fiscal_quarter": "Q1",
        "cik": "0000037996"
    },
    # =========================================================================
    # ADDITIONAL TEMPORAL DATA (older filings for trend analysis)
    # =========================================================================
    {
        "content": (
            "Total revenue for fiscal year 2022 was $81.46 billion, an increase "
            "of 51% compared to $53.82 billion for fiscal year 2021. We "
            "delivered approximately 1.31 million vehicles during 2022, compared "
            "to 936,222 vehicles during 2021, an increase of 40%. Automotive "
            "gross margin was 25.6%, reflecting strong pricing power and "
            "manufacturing efficiencies, though offset partially by raw material "
            "cost increases and supply chain challenges."
        ),
        "company": "Tesla Inc.",
        "ticker": "TSLA",
        "filing_type": "10-K",
        "section": "Management Discussion and Analysis",
        "filing_date": "2023-01-31",
        "fiscal_year": "2022",
        "fiscal_quarter": "FY",
        "cik": "0001318605"
    },
    {
        "content": (
            "The ongoing global semiconductor shortage significantly impacted "
            "our production capabilities in 2022. We were unable to fully meet "
            "customer demand for several vehicle lines due to chip shortages, "
            "resulting in estimated lost production of approximately 400,000 "
            "vehicles. We are actively working with semiconductor suppliers to "
            "secure additional capacity and are redesigning certain components "
            "to use more readily available chips."
        ),
        "company": "Ford Motor Company",
        "ticker": "F",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2023-02-02",
        "fiscal_year": "2022",
        "fiscal_quarter": "FY",
        "cik": "0000037996"
    },
    {
        "content": (
            "Microsoft Cloud revenue grew 22% to $111.6 billion in fiscal 2023. "
            "Azure and other cloud services revenue grew 29%, with AI services "
            "contributing an accelerating portion of growth. We announced our "
            "partnership with OpenAI and the integration of AI capabilities "
            "across our product portfolio, including GitHub Copilot, Microsoft "
            "365 Copilot, and Azure OpenAI Service. These investments represent "
            "a strategic bet on AI as the next computing platform."
        ),
        "company": "Microsoft Corporation",
        "ticker": "MSFT",
        "filing_type": "10-K",
        "section": "Management Discussion and Analysis",
        "filing_date": "2023-07-27",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0000789019"
    },
    # =========================================================================
    # FINANCIAL QA BENCHMARK DATA
    # =========================================================================
    {
        "content": (
            "Net interest income is the difference between interest earned on "
            "assets, such as loans and securities, and interest paid on "
            "liabilities, such as deposits and borrowings. For JPMorgan Chase, "
            "net interest income was $89.3 billion in 2023, a 34% increase "
            "from $66.7 billion in 2022, driven primarily by higher interest "
            "rates. Net interest margin (NIM) expanded to 2.72% in 2023 from "
            "2.16% in 2022, benefiting from the Federal Reserve's rate hiking "
            "cycle which began in March 2022."
        ),
        "company": "JPMorgan Chase & Co.",
        "ticker": "JPM",
        "filing_type": "10-K",
        "section": "Financial Statements",
        "filing_date": "2024-02-16",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0000019617"
    },
    {
        "content": (
            "Our competitive position in the electric vehicle market continues "
            "to evolve. Global EV competition has intensified significantly, "
            "particularly from Chinese manufacturers such as BYD, which "
            "surpassed Tesla in total EV sales volume in Q4 2023. We continue "
            "to compete on the basis of product performance, technology, brand, "
            "charging infrastructure (Supercharger network), manufacturing "
            "efficiency, and total cost of ownership. Our next-generation "
            "vehicle platform, expected to begin production in H2 2025, aims "
            "to achieve a 50% reduction in manufacturing costs."
        ),
        "company": "Tesla Inc.",
        "ticker": "TSLA",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-01-29",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0001318605"
    },
    {
        "content": (
            "Climate-related risks are increasingly material to our operations. "
            "Physical risks including extreme weather events, sea-level rise, "
            "and temperature changes may disrupt our manufacturing facilities "
            "and supply chain. Transition risks include changing consumer "
            "preferences toward electric vehicles, evolving emissions "
            "regulations, and potential carbon pricing mechanisms. We have "
            "committed to achieving carbon neutrality across our global "
            "operations by 2035 and across our entire value chain by 2050."
        ),
        "company": "Ford Motor Company",
        "ticker": "F",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2024-02-06",
        "fiscal_year": "2023",
        "fiscal_quarter": "FY",
        "cik": "0000037996"
    },
]


# Pre-built evaluation QA pairs for RAGAS benchmarking
EVALUATION_QA_PAIRS = [
    {
        "question": "What was Tesla's total revenue in 2023?",
        "ground_truth": "Tesla's total revenue for fiscal year 2023 was $96.77 billion.",
        "expected_context_company": "Tesla Inc."
    },
    {
        "question": "How much did Ford lose on electric vehicles in 2023?",
        "ground_truth": "Ford Model e reported an EBIT loss of $4.7 billion in 2023.",
        "expected_context_company": "Ford Motor Company"
    },
    {
        "question": "What was Microsoft's Azure revenue growth rate?",
        "ground_truth": "Azure and other cloud services revenue grew 30% in fiscal year 2024.",
        "expected_context_company": "Microsoft Corporation"
    },
    {
        "question": "What supply chain risks does Tesla face?",
        "ground_truth": "Tesla faces risks related to availability and cost of raw materials including lithium, nickel, cobalt, and manganese used in battery cells.",
        "expected_context_company": "Tesla Inc."
    },
    {
        "question": "What cybersecurity incident did Microsoft disclose?",
        "ground_truth": "In January 2024, Microsoft disclosed that a nation-state actor (Midnight Blizzard) had accessed certain executive email accounts.",
        "expected_context_company": "Microsoft Corporation"
    },
    {
        "question": "What was Apple's services revenue in 2024?",
        "ground_truth": "Apple's services revenue grew 13% to $92.9 billion in fiscal year 2024.",
        "expected_context_company": "Apple Inc."
    },
    {
        "question": "How many vehicles did Tesla deliver in Q3 2024?",
        "ground_truth": "Tesla delivered approximately 462,890 vehicles during Q3 2024.",
        "expected_context_company": "Tesla Inc."
    },
    {
        "question": "What is JPMorgan's technology budget?",
        "ground_truth": "JPMorgan Chase had a technology budget of approximately $15.8 billion in 2023.",
        "expected_context_company": "JPMorgan Chase & Co."
    },
    {
        "question": "What was NVIDIA's data center revenue growth?",
        "ground_truth": "NVIDIA's Data Center revenue was $47.5 billion in fiscal 2024, an increase of 217%.",
        "expected_context_company": "NVIDIA Corporation"
    },
    {
        "question": "How is Ford restructuring its EV strategy?",
        "ground_truth": "Ford is reallocating approximately $12 billion in planned EV investments, delaying certain next-generation EV programs and shifting capital toward hybrid vehicle development.",
        "expected_context_company": "Ford Motor Company"
    },
]


def get_all_documents() -> List[Dict[str, Any]]:
    """Return all sample documents."""
    return SAMPLE_DOCUMENTS.copy()


def get_documents_by_company(company_name: str) -> List[Dict[str, Any]]:
    """Filter documents by company name (partial match)."""
    return [
        doc for doc in SAMPLE_DOCUMENTS
        if company_name.lower() in doc["company"].lower()
        or company_name.lower() in doc.get("ticker", "").lower()
    ]


def get_documents_by_type(filing_type: str) -> List[Dict[str, Any]]:
    """Filter documents by filing type."""
    return [
        doc for doc in SAMPLE_DOCUMENTS
        if filing_type.upper() in doc["filing_type"].upper()
    ]


def get_companies() -> List[str]:
    """Get unique company names in the dataset."""
    return list(set(doc["company"] for doc in SAMPLE_DOCUMENTS))


def get_tickers() -> List[str]:
    """Get unique tickers in the dataset."""
    return list(set(doc.get("ticker", "") for doc in SAMPLE_DOCUMENTS if doc.get("ticker")))


def get_evaluation_pairs() -> List[Dict[str, Any]]:
    """Return evaluation QA pairs for benchmarking."""
    return EVALUATION_QA_PAIRS.copy()
