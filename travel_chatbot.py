import os
from dotenv import load_dotenv
from functools import lru_cache
from crewai import LLM, Agent, Task, Crew, Process
from datetime import datetime
import requests
import json
from crewai.tools import tool          # decorator
from crewai_tools import SerperDevTool # web-search tool
from IPython.display import Markdown, display
import re

# Load environment variables from .env file
load_dotenv()

# Set the environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI1_API_KEY = os.getenv("GEMINI1_API_KEY")

GEMINIPRO_API_KEY = os.getenv("GEMINIPRO_API_KEY")

OPENROUTER_API_KEY3=os.getenv("OPENROUTER_API_KEY3")
OPENAI_API_BASE=os.getenv("OPENAI_API_BASE")

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

print("API Keys loaded successfully.")

@lru_cache(maxsize=1)
def initialize_llm():
    return LLM(
        model="openrouter/z-ai/glm-4.5-air:free",
        api_key=OPENROUTER_API_KEY3,
        base_url=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        temperature=0.4,        # lower randomness for agentic use            # enable streaming if helpful
    )

@lru_cache(maxsize=1)
def initialize_llm1():
    """Initialize and cache the LLM instance to avoid repeated initializations."""
    return LLM(
        model="gemini/gemini-2.0-flash",
        provider="google",
        api_key=GEMINI_API_KEY
    )

@lru_cache(maxsize=1)
def initialize_llmPro():
    """Initialize and cache the LLM instance to avoid repeated initializations."""
    return LLM(
        model="gemini/gemini-2.5-flash",
        provider="google",
        api_key=GEMINIPRO_API_KEY
    )   



# Initialize the web search tool
search_tool = SerperDevTool()

# Tool 1: Human Input Tool
# This tool pauses the execution and asks for human input.
@tool("Human Input Tool")
def human_input_tool(question: str) -> str:
    """Asks a human for input. Returns only the user's response without additional context."""
    # Clear any pending output and ensure the prompt is visible
    print("\n" + "="*50)
    print("HUMAN INPUT REQUIRED")
    print("="*50)
    print(f"\n{question}\n")
    print("="*50)
    
    # Flush stdout to ensure the prompt is displayed before waiting for input
    import sys
    sys.stdout.flush()
    
    # Get user input
    user_response = input("\nYour response: ")
    
    # Clean and return just the user's input
    return user_response.strip()

# Enhanced date parsing function
def parse_flexible_dates(date_input: str) -> str:
    """Convert flexible date formats to YYYY-MM-DD format"""
    if not date_input or date_input.lower() in ['flexible', 'no preferred date', 'any time']:
        return 'flexible'
    
    # Try to parse common date formats
    import re
    from datetime import datetime
    
    # Handle formats like "august 5th to 6th", "aug 5 to 6", etc.
    current_year = datetime.now().year
    
    # Pattern for "month day to day" format
    pattern1 = r'(\w+)\s+(\d+)(?:st|nd|rd|th)?\s+to\s+(\d+)(?:st|nd|rd|th)?'
    match1 = re.search(pattern1, date_input.lower())
    
    if match1:
        month_name, start_day, end_day = match1.groups()
        try:
            # Convert month name to number
            month_map = {
                'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
                'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
                'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
                'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
            }
            
            month_num = month_map.get(month_name)
            if month_num:
                start_date = f"{current_year}-{month_num:02d}-{int(start_day):02d}"
                end_date = f"{current_year}-{month_num:02d}-{int(end_day):02d}"
                return f"{start_date} to {end_date}"
        except:
            pass
    
    # If parsing fails, return the original input
    return date_input

def geocode_city(city: str) -> tuple[float, float] | None:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    resp = requests.get(url, params={"name": city, "count": 1, "language": "en"})
    resp.raise_for_status()
    results = resp.json().get("results")
    if results:
        return results[0]["latitude"], results[0]["longitude"]
    return None

# Tool 2: Weather Tool (Updated for Forecast)
bad_weather_codes = [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99]
desc_map = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Fog depositing rime",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

@tool("Weather Tool")
def open_meteo_weather_tool(city: str, start_date: str, end_date: str) -> str:
    """Returns weather forecast for a city between start_date and end_date using Open-Meteo."""
    coords = geocode_city(city)
    if not coords:
        return f"Sorry, I couldn’t find coordinates for {city}."
    lat, lon = coords
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,weathercode",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto"
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        daily = data["daily"]
        forecast_lines = [f"Weather forecast for {city.title()} from {start_date} to {end_date}:"]
        bad_weather_dates = []
        for i in range(len(daily["time"])):
            date = daily["time"][i]
            max_temp = daily["temperature_2m_max"][i]
            min_temp = daily["temperature_2m_min"][i]
            code = daily["weathercode"][i]
            desc = desc_map.get(code, "unknown")
            forecast_lines.append(f"- {date}: {min_temp}°C to {max_temp}°C, {desc}")
            if code in bad_weather_codes:
                bad_weather_dates.append(date)
        if bad_weather_dates:
            forecast_lines.append("\nNote: Bad weather (rain, snow, or thunderstorms) expected on: " + ", ".join(bad_weather_dates))
        return "\n".join(forecast_lines)
    except Exception as e:
        return f"Error fetching Open-Meteo data: {e}"

# Tool 3: Currency Conversion Tool
def get_conversion_rate(from_currency: str, to_currency: str) -> float | None:
    """Helper function to get a numerical conversion rate."""
    try:
        url = f"https://open.er-api.com/v6/latest/{from_currency}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['rates'][to_currency]
    except Exception:
        return None

# Your existing tool can now be simplified
@tool("Currency Conversion Tool")
def currency_conversion_tool(from_currency: str, to_currency: str, amount: str = "1") -> str:
    """
    Returns the conversion rate from one currency to another, or converts a specific amount.
    
    Args:
        from_currency: The source currency code (e.g., "USD")
        to_currency: The target currency code (e.g., "LKR")
        amount: The amount to convert (default: "1" for just the rate)
    
    Returns:
        A string with the conversion result in JSON format: {"rate": X.XXXX, "converted_amount": X.XX}
    """
    try:
        rate = get_conversion_rate(from_currency, to_currency)
        if rate:
            amount_float = float(amount)
            converted_amount = amount_float * rate
            return json.dumps({
                "rate": rate,
                "converted_amount": converted_amount
            })
        return json.dumps({"error": f"Error converting currency. Ensure currency codes are correct."})
    except Exception as e:
        return json.dumps({"error": f"Error converting currency: {str(e)}"})

def format_currency(amount: float, currency_code: str) -> str:
    """Format a currency amount with the appropriate number of decimal places and separators."""
    # Most currencies use 2 decimal places
    decimal_places = 2
    
    # Some currencies don't use decimal places
    no_decimal_currencies = ["JPY", "KRW", "VND", "IDR", "CLP", "PYG", "HUF"]
    if currency_code in no_decimal_currencies:
        decimal_places = 0
    
    # Format with thousand separators and appropriate decimal places
    if decimal_places == 0:
        formatted = f"{int(round(amount)):,}"
    else:
        formatted = f"{amount:,.{decimal_places}f}"
    
    return f"{formatted} {currency_code}"

print("Tools created successfully.")

# Define country to currency mapping
country_to_currency = {
    'Australia': 'AUD',
    'Brazil': 'BRL',
    'Canada': 'CAD',
    'China': 'CNY',
    'France': 'EUR',
    'Germany': 'EUR',
    'India': 'INR',
    'Italy': 'EUR',
    'Japan': 'JPY',
    'Mexico': 'MXN',
    'Singapore': 'SGD',
    'South Africa': 'ZAR',
    'Spain': 'EUR',
    'Sri Lanka': 'LKR',
    'Switzerland': 'CHF',
    'Thailand': 'THB',
    'United Arab Emirates': 'AED',
    'United Kingdom': 'GBP',
    'United States': 'USD',
    # Add more as needed
}

def parse_budget_from_text(text: str) -> str:
    """
    Enhanced budget parser that handles various natural language formats.
    Returns budget in "AMOUNT CURRENCY_CODE" format or "null" if not found.
    """
    
    # Currency mappings - both full names and codes
    currency_mappings = {
        # Full currency names to codes
        'sri lankan rupees': 'LKR',
        'sri lankan rupee': 'LKR',
        'lankan rupees': 'LKR',
        'rupees': 'LKR',  # Default rupees to LKR unless context suggests otherwise
        'indian rupees': 'INR', 
        'indian rupee': 'INR',
        'us dollars': 'USD',
        'us dollar': 'USD',
        'american dollars': 'USD',
        'dollars': 'USD',
        'british pounds': 'GBP',
        'pounds sterling': 'GBP',
        'pounds': 'GBP',
        'euros': 'EUR',
        'euro': 'EUR',
        'japanese yen': 'JPY',
        'yen': 'JPY',
        'thai baht': 'THB',
        'baht': 'THB',
        'australian dollars': 'AUD',
        'canadian dollars': 'CAD',
        'singapore dollars': 'SGD',
        'swiss francs': 'CHF',
        'south african rand': 'ZAR',
        'rand': 'ZAR',
        
        # Currency codes (already in correct format)
        'lkr': 'LKR',
        'inr': 'INR', 
        'usd': 'USD',
        'gbp': 'GBP',
        'eur': 'EUR',
        'jpy': 'JPY',
        'thb': 'THB',
        'aud': 'AUD',
        'cad': 'CAD',
        'sgd': 'SGD',
        'chf': 'CHF',
        'zar': 'ZAR',
    }
    
    # Clean and normalize the text
    text_lower = text.lower().strip()
    
    # Remove commas and extra spaces from numbers
    text_normalized = re.sub(r'(\d+)\s*,?\s*(\d+)', r'\1\2', text_lower)
    
    # Pattern 1: "budget is X currency" or "X currency budget"
    patterns = [
        # "budget is 50000 sri lankan rupees"
        r'budget\s+is\s+(\d+(?:\s*,?\s*\d+)*)\s+(.+?)(?:\s|$)',
        # "50000 sri lankan rupees budget" or "50000 sri lankan rupees"
        r'(\d+(?:\s*,?\s*\d+)*)\s+(.+?)\s*(?:budget|$)',
        # "budget of 50000 indian rupees"
        r'budget\s+of\s+(\d+(?:\s*,?\s*\d+)*)\s+(.+?)(?:\s|$)',
        # "my budget is 50000 rupees"
        r'my\s+budget\s+is\s+(\d+(?:\s*,?\s*\d+)*)\s+(.+?)(?:\s|$)',
        # "the budget is 50000 LKR"
        r'the\s+budget\s+is\s+(\d+(?:\s*,?\s*\d+)*)\s+(.+?)(?:\s|$)',
        # Just "50000 LKR" format
        r'(\d+(?:\s*,?\s*\d+)*)\s+([a-z]{3})(?:\s|$)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text_normalized)
        for match in matches:
            amount_str = match.group(1)
            currency_str = match.group(2).strip()
            
            # Clean the amount (remove spaces and commas)
            amount_clean = re.sub(r'[,\s]', '', amount_str)
            
            # Try to match the currency
            currency_code = None
            
            # Direct lookup
            if currency_str in currency_mappings:
                currency_code = currency_mappings[currency_str]
            else:
                # Fuzzy matching for partial matches
                for currency_name, code in currency_mappings.items():
                    if currency_name in currency_str or currency_str in currency_name:
                        currency_code = code
                        break
            
            if currency_code and amount_clean.isdigit():
                return f"{amount_clean} {currency_code}"
    
    # Pattern 2: Look for currency symbols
    symbol_patterns = [
        (r'\$(\d+(?:,\d+)*)', 'USD'),  # $50000
        (r'₹(\d+(?:,\d+)*)', 'INR'),   # ₹50000
        (r'£(\d+(?:,\d+)*)', 'GBP'),   # £50000
        (r'€(\d+(?:,\d+)*)', 'EUR'),   # €50000
    ]
    
    for pattern, currency in symbol_patterns:
        match = re.search(pattern, text)
        if match:
            amount = re.sub(r'[,\s]', '', match.group(1))
            return f"{amount} {currency}"
    
    return "null"

def calculate_nights(dates: str) -> int:
    """Calculates the number of nights for a given date range."""
    try:
        start_str, end_str = dates.split(' to ')
        start_date = datetime.strptime(start_str.strip(), '%Y-%m-%d')
        end_date = datetime.strptime(end_str.strip(), '%Y-%m-%d')
        # The number of nights is the difference in days
        num_nights = (end_date - start_date).days
        return max(0, num_nights)  # Return 0 if dates are invalid or same day
    except (ValueError, IndexError):
        return 0

def extract_json_from_response(response_text: str) -> dict:
    """
    Extract and parse JSON from agent response that might contain markdown formatting
    or extra text around the JSON.
    """
    try:
        # First, try to parse as-is (in case it's clean JSON)
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Remove markdown code blocks
    cleaned_text = response_text.strip()
    
    # Remove ```json and ``` markers
    if '```json' in cleaned_text:
        # Extract content between ```json and ```
        pattern = r'```json\s*(.*?)\s*```'
        match = re.search(pattern, cleaned_text, re.DOTALL)
        if match:
            cleaned_text = match.group(1)
    elif '```' in cleaned_text:
        # Extract content between ``` markers
        pattern = r'```\s*(.*?)\s*```'
        match = re.search(pattern, cleaned_text, re.DOTALL)
        if match:
            cleaned_text = match.group(1)
    
    # Remove any leading/trailing text that's not JSON
    # Look for the first { and last }
    start_idx = cleaned_text.find('{')
    end_idx = cleaned_text.rfind('}')
    
    if start_idx != -1 and end_idx != -1:
        cleaned_text = cleaned_text[start_idx:end_idx+1]
    
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Attempted to parse: {cleaned_text}")
        raise        

def create_setup_crew(initial_prompt: str):
    """Creates the crew responsible for gathering user requirements."""
    llmpro = initialize_llmPro() # Use a fast and reliable LLM for conversation

    current_date = datetime.now().strftime('%Y-%m-%d')

    # This agent's job is to talk to the user and fill out a form.
    setup_agent = Agent(
        role="Trip Requirements Specialist",
        goal="Accurately capture all necessary details for a travel itinerary from the user. "
             "Your final goal is to produce a JSON object with all the required information.",
        backstory="You are a friendly and efficient assistant who helps users plan their "
                  "dream vacation. You are programmed to ask clarifying questions one by one "
                  "until you have all the information needed to create a travel plan.",
        tools=[human_input_tool],
        llm=llmpro,
        verbose=False
    )

    # Pre-parse the budget using our enhanced function
    parsed_budget = parse_budget_from_text(initial_prompt)

    # Enhanced task with better date handling
    setup_task = Task(
        description=f"""
        **CRITICAL: You must start by analyzing the initial user prompt and extracting ALL available information BEFORE asking any questions.**

        **Initial User Prompt:** "{initial_prompt}"

        **STEP 1 - ANALYZE AND EXTRACT (DO THIS FIRST):**
        Carefully read the initial prompt above and extract information to pre-fill this JSON:
        {{
          "location": "null",
          "interests": "null", 
          "budget": "null",
          "num_people": "null",
          "travel_dates": "null",
          "preferred_currency": "null"
        }}

        From the prompt "{initial_prompt}", you should be able to extract:
        - Location: Look for place names (e.g., "mirissa" → "Mirissa, Sri Lanka")
        - Number of people: Look for numbers + "people" (e.g., "8 people" → "8")  
        - Interests: Look for activities/preferences and additional details in the prompt. Get the whole portion of the prompt related to interests (eg: "I want to go to mirissa. i would like a villa with a pool. and some clubbing in the night. and also i want to explore mirissa. make the food as cheap as possible and save my budget for other stuff. This is for 5 people and the budget is 50000 LKR: i would like a villa with a pool. and some clubbing in the night. and also i want to explore mirissa. make the food as cheap as possible and save my budget for other stuff")
        - Budget: Look for money amounts in the format "AMOUNT CURRENCY" (e.g., "50000 LKR")
        - **Travel Dates**: Look for date references in ANY format:
           * "september 6th and come back on 7th september" → "2025-09-06 to 2025-09-07"
           * "august 5th to 6th" → "2025-08-05 to 2025-08-06"
           * "travel on [date] and come back on [date]" → convert to YYYY-MM-DD format
           * Always assume year 2025 unless specified otherwise
           * If dates indicate flexibility (like "no preferred date", "flexible", "any time"), set to "flexible"
        - Currency preferences: Look for currency mentions

        **CRITICAL BUDGET EXTRACTION INSTRUCTIONS:**
        - When extracting budget, you MUST include both the amount AND the currency code
        - For example, if the prompt says "budget is 50000 LKR", the budget value should be "50000 LKR"
        - Do NOT separate the amount and currency - they must be together in one string
        - The budget format must be "AMOUNT CURRENCY" (e.g., "50000 LKR", "250 USD")
        
        **CRITICAL DATE EXTRACTION EXAMPLES:**
        - "planning to travel on september 6th and come back on 7th september" → "2025-09-06 to 2025-09-07"
        - planning to travel from september 6th to 7th september" → "2025-09-06 to 2025-09-07"
        - "going from august 15 to august 20" → "2025-08-15 to 2025-08-20"
        - "travel in december 25th to 28th" → "2025-12-25 to 2025-12-28"

        **STEP 2 - ASK FOR MISSING INFO ONLY:**
        After pre-filling from the prompt, follow these rules:
        
        - **For budget**: If still "null", ask the user for their budget
        - **For travel_dates**: If still "null", ask the user: "What are your preferred travel dates? (You can say 'flexible' if you don't have specific dates)"
          
          **IMPORTANT DATE HANDLING:**
          - If the user responds with anything indicating flexibility (like "no preferred date", "flexible", "just checking casually", "any time", etc.), set travel_dates to "flexible"
          - If they provide dates in casual format, you must convert them to proper YYYY-MM-DD format
          - Always assume current year (2025) unless specified otherwise
          
        - **For preferred_currency**: If still "null", DO NOT ASK the user. Instead, automatically determine the local currency based on the destination country:
          * Sri Lanka → "LKR"
          * India → "INR" 
          * Thailand → "THB"
          * United States → "USD"
          * United Kingdom → "GBP"
          * European countries (France, Germany, Italy, Spain) → "EUR"
          * Japan → "JPY"
          * Australia → "AUD"
          * Canada → "CAD"
          * Singapore → "SGD"
          * Default to "USD" if country not recognized
        
        Ask one question at a time using the Human Input Tool. Skip asking about preferred_currency entirely.

        **STEP 3 - FINAL OUTPUT FORMAT:**
        **CRITICAL**: Your final answer must be EXACTLY a single JSON object with NO markdown formatting, NO code blocks, NO backticks, and NO additional text.

        Example of CORRECT format:
        {{"location": "Mirissa, Sri Lanka", "interests": "i would like a villa with a pool. and some clubbing in the night. and also i want to explore mirissa. make the food as cheap as possible and save my budget for other stuff", "budget": "50000 LKR", "num_people": "5", "travel_dates": "2025-08-05 to 2025-08-06", "preferred_currency": "LKR"}}

        **WRONG formats (DO NOT USE):**
        - ```json {{ ... }} ```
        - Here is the JSON: {{ ... }}
        - The details are: {{ ... }}

        **IMPORTANT:** 
        - Current date is {current_date}
        - Your final answer must be ONLY a valid JSON string, nothing else
        - Do not include any explanatory text in your final answer
        - Do not use markdown code blocks
        """,
        expected_output="A single, valid JSON string containing all the extracted and gathered travel details WITHOUT any markdown formatting.",
        agent=setup_agent
    )

    return Crew(
        agents=[setup_agent],
        tasks=[setup_task],
        verbose=False
    )

def invoke_agent(location, interests, budget, num_people, travel_dates, preferred_currency):
    """Invokes the travel agent with the given inputs."""

    budget_in_usd = float('inf') # Default to infinite budget if flexible
    budget_instruction = "The user has not specified a budget. Suggest a range of options from budget-friendly to luxury."
    
    if budget.lower() != 'flexible':
        try:
            budget_amount_str, budget_currency = budget.strip().split()
            budget_amount = float(budget_amount_str)
            budget_currency = budget_currency.upper()
        except ValueError:
            print("Error: Invalid budget format. Please use 'AMOUNT CURRENCY' (e.g., '250 USD').")
            return

        budget_in_usd = budget_amount
        if budget_currency != 'USD':
            rate_to_usd = get_conversion_rate(budget_currency, 'USD')
            if rate_to_usd:
                budget_in_usd = budget_amount * rate_to_usd
        
        budget_instruction = f"The total available budget is {budget_in_usd:.2f} USD. All suggested activities and accommodation must fit within this budget and should be **CLOSE** and MUST BE LESS THAN OR EQUAL to the budget."

    # Determine local currency
    country = location.split(',')[-1].strip()
    local_currency = country_to_currency.get(country, 'USD')
    target_currency = preferred_currency if preferred_currency else local_currency

    # Determine if accommodation is needed
    # Calculate the number of nights
    num_nights = 0
    accommodation_instruction = ""
    weather_tool_usage_instruction = "The user has not provided specific travel dates or wants flexible dates. You cannot use the Weather Tool. Instead, provide general advice about the best seasons to visit."
    
    if travel_dates.lower() != 'flexible':
        num_nights = calculate_nights(travel_dates)
        weather_tool_usage_instruction = f"You MUST use the Weather Tool with the exact start and end dates: {travel_dates}."
        if num_nights > 0:
            accommodation_instruction = f"**Crucially, you MUST research and suggest one suitable accommodation for a {num_nights}-night stay.**"
    else:
        # If dates are flexible, you might ask for accommodation for a default number of nights, like 3.
        accommodation_instruction = "**Since dates are flexible, you can optionally suggest one accommodation suitable for a 2-3 night stay as an example.**"

    # Initialize LLM
    llm_model = initialize_llm()
    llm1_model = initialize_llm1()



    # Agent 1: Local Data Agent
    local_data_agent = Agent(
        role="Local Data Specialist",
        goal="Fetch weather and currency data for the travel destination.",
        backstory="An analyst providing real-time travel insights.",
        tools=[open_meteo_weather_tool, currency_conversion_tool],
        llm=llm1_model,
        verbose=False
    )

    # Agent 2: Web Search Agent (City Expert)
    city_expert_agent = Agent(
        role='Expert City Researcher',
        goal='Efficiently find a specific number of activities and accommodation within a budget.',
        backstory='A travel enthusiast who finds the best spots tailored to your needs, focusing on speed and accuracy.',
        tools=[search_tool],
        llm=llm_model,
        verbose=False,
        max_iter=15,  # Hard limit on the number of execution loops (thinking -> tool -> observation)
        allow_delegation=False
    )

    # Agent 3: Budget Verifier Agent
    budget_verifier_agent = Agent(
        role='Budget Verification Analyst',
        goal='Critically analyze the researched activities and their estimated costs against the user-provided budget. Provide a clear "go" or "no-go" verdict with justification.',
        backstory='A meticulous financial analyst with a knack for sniffing out hidden costs and ensuring travel plans are financially sound. You are firm but fair.',
        tools=[],
        llm=llm1_model,
        allow_delegation=False,
        verbose=False
    )

    # Agent 4: Travel Concierge Agent
    travel_concierge_agent = Agent(
        role='Head Travel Concierge',
        goal='Synthesize all gathered information into a cohesive, beautifully formatted travel itinerary with weather insights and converted costs.',
        backstory='A world-class concierge from a five-star hotel, known for creating personalized and delightful travel experiences. You are meticulous about financial accuracy and ensure all currency conversions are precise and consistent',
        tools=[currency_conversion_tool],  # Added for cost conversion
        llm=llm1_model,
        allow_delegation=False,
        verbose=False
    )

    print("Agents defined successfully.")

    # Task 1: Get local data (weather forecast and currency conversion)
    task_get_local_data = Task(
        description=f"""Fetch the currency conversion rate from USD to the local currency for {location}.
        {weather_tool_usage_instruction}
        """,
        expected_output="A summary of the weather forecast for the specified dates and the USD to local currency conversion rate.",
        agent=local_data_agent
    )

    # Task 2: Find city information
    task_find_city_info = Task(
        description=f"""
        For a group of {num_people} people traveling to {location} with interests in '{interests}'.

        **TRAVEL DATES:** {travel_dates}

        {budget_instruction}
        {accommodation_instruction}

        **IMPORTANT CONTEXT USAGE:** You will receive context from a data specialist that includes a real-time currency conversion rate. If you find prices online in a local currency (e.g., INR, LKR), you **must use the precise conversion rate provided in your context** to convert them to USD for your analysis and final JSON output. This is more accurate than using your general knowledge.

        **CRITICAL INSTRUCTION**: For each item you research (especially accommodation, restaurants, and specific activities), you MUST use the search tool to find a relevant webpage (like a booking page, official website, or Google Maps link) and include it in the 'link' field of your JSON output. If no direct link is available, you can set the value to "null".
        The TOTAL estimated cost of all researched items (in USD) must not exceed this budget and also should be close to this budget.

        **Your instructions are to be highly efficient. Aim to use the web search tool no more than 2-3 times.**

        Your research output MUST contain the following specific items:
        1.   Search for the best options that match with the interests and the budget. **YOU MUST make sure your search includes 3 meals (breakfast, lunch, dinner) per day and optionally a dinner on the last day of the trip.**
        2.  {accommodation_instruction} 

        Your final answer MUST be a single JSON string. This JSON object should contain a key "items" which is a list of dictionaries, and a key "total_estimated_cost_usd".
        Each dictionary in the "items" list must have the keys: "type" (string, e.g., "accommodation" or "activity"), "name" (string), "description" (string), "cost_usd" (number), and "link" (string or null).
        """,
        expected_output="""A single, valid JSON string that can be directly parsed. Example format: 
        '{"items": [{"type": "accommodation", "name": "Mirissa Beach Villa", "description": "A beautiful villa with a pool for 4 guests.", "cost_usd": 150, "link": "https://example.com/villa"}, {"type": "activity", "name": "Whale Watching Tour", "description": "A 4-hour whale watching excursion.", "cost_usd": 80, "link": "https://example.com/whale-watching"}], "total_estimated_cost_usd": 230}'
        """,
        agent=city_expert_agent,
        context=[task_get_local_data]
    )

    # Task 3: Verify the budget
    task_verify_budget = Task(
        description=f"""Analyze the research from the city expert.
        {budget_instruction}
        Sum up the total estimated cost of ALL items (activities and accommodation) provided by the researcher.Compare this total to the available USD budget. Provide a clear 'go' or 'no-go' verdict with a brief justification. The user's original budget was '{budget_in_usd}'.""",
        expected_output="A budget feasibility verdict (Go/No-Go) comparing the total estimated cost in USD against the total available budget in USD.",
        agent=budget_verifier_agent,
        context=[task_find_city_info]
    )

    # Task 4: Compile the final report
    task_compile_report = Task(
        description=f"""
        Create a final, human-readable travel itinerary for {num_people} people for a trip to {location}.

        **TRAVEL DATES:** {travel_dates}
        
        **Handle flexible dates:** If travel dates are "flexible", mention this prominently and suggest the best seasons to visit {location} with reasons (weather, prices, crowds, etc.).

        You will receive structured data in a JSON string format from the city expert's context. Your first step is to parse this JSON to access the list of activities and accommodation.

        Your report must:
        1.  First, use the Currency Conversion Tool ONCE to get the numerical conversion rate from USD to {target_currency}.
        2.  Parse the JSON response from the tool to extract the exact conversion rate.
        3.  Store this rate and use it consistently for ALL currency conversions in your report.
        4.  For each item in the parsed JSON:
            a. Extract the 'cost_usd' value
            b. Multiply it by the stored conversion rate to get the exact amount in {target_currency}
            c. Format the result as "X,XXX.XX {target_currency}" (with appropriate decimal places)
            d. **When displaying the cost, show ONLY the final converted amount. Do NOT show the original USD cost or the mathematical calculation used to arrive at the final price.**
               For example, instead of writing "Cost: 100 USD x 301.95 = 30,195 LKR", you MUST write "Cost: 30,195 LKR"..
            e. **If the item has a 'link' that is not "null", you MUST format it as a clickable markdown link.**
               For example, if the name is "Mirissa Beach Villa" and the link is "https://example.com/villa", you must provide the clickable link near the name.
        5.  For every activity/ meal (eg: breakfast, lunch, dunner)/  scenary or literally anything, **YOU MUST mention the cost if the user has to pay for it**.   
        6.  Synthesize the parsed items into a cohesive, daily plan.
        7.  **Important:** Do NOT display the 'USD to {target_currency}' conversion rate in the report if the user's original budget was already provided in {target_currency}. Only show the conversion rate if the original budget currency was different from the final report currency.
        8.  Incorporate the budget verification verdict from the context.
        9.  Include the weather insights if available. If specific weather data was fetched, incorporate it. If dates are flexible, provide seasonal recommendations instead.
        10.  At the end of the report, give a budget summary of the total cost of the trip in {target_currency}.
        11.  Format the entire output as a beautiful and exciting markdown report. Display all final costs ONLY in {target_currency}.
        """,
        expected_output=f"A complete, beautifully formatted markdown report with a travel plan, budget analysis, and weather/seasonal insights. All costs must be in {target_currency} and must not show any calculations.",
        agent=travel_concierge_agent,
        context=[task_verify_budget, task_get_local_data, task_find_city_info]
    )

    print("Tasks created successfully.")



    # Create the Crew
    travel_crew = Crew(
        agents=[local_data_agent, city_expert_agent, budget_verifier_agent, travel_concierge_agent],
        tasks=[task_get_local_data, task_find_city_info, task_verify_budget, task_compile_report],
        process=Process.sequential,
        verbose=False
    )

    # Kick off the crew's work!
    result = travel_crew.kickoff()


    if hasattr(result, 'raw') and isinstance(result.raw, str):
        return result
    else:
        # If the result is not in the expected format, return an error string
        print(f"Error: Unexpected result format. Type: {type(result)}, Value: {result}")
        return "Error: The travel agent returned an unexpected result format."

# invoke_agent("Mirissa, Sri Lanka", "entertainment, beach and affordable villa with pool. we need lunch dinner breakfast to eat in an affordable way", "32000 LKR", 4, "2025-08-05 to 2025-08-06", "")

def run_travel_chatbot(initial_prompt=None):
    """The main entry point for the conversational travel agent."""
    
    # 1. Get initial prompt from the user if not provided
    if initial_prompt is None:
        print("Welcome to the Travel Planning Assistant!")
        print("Please describe your travel plans (destination, dates, interests, budget, etc.):")
        initial_prompt = input("> ")
    
    # 2. Run the Setup Crew to gather all details - PASS THE PROMPT DIRECTLY
    setup_crew = create_setup_crew(initial_prompt)
    
    # The kickoff method returns a CrewOutput object, not a string
    trip_details_output = setup_crew.kickoff()
    
    print("\n--- Trip Details Gathered ---")
    # Access the raw string from the CrewOutput object using the .raw attribute
    trip_details_json_str = trip_details_output.raw 
    print(trip_details_json_str)
    print("---------------------------\n")
    try:
        # 3. Parse the details and run the main planning crew
        details = extract_json_from_response(trip_details_json_str)
        
        invoke_agent(
            location=details['location'],
            interests=details['interests'],
            budget=details['budget'],
            num_people=details['num_people'],
            travel_dates=details['travel_dates'],
            preferred_currency=details['preferred_currency']
        )
    except (json.JSONDecodeError, KeyError) as e:
        print(f"\nSorry, there was an error processing the details: {e}")
        print("Let's try again.")

# --- Replace your old script execution call with this ---
if __name__ == "__main__":
    run_travel_chatbot()
