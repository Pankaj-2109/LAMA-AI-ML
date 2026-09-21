"""
Seed datasets for LAMA AI ML Pipeline.
Covers:
1. Intent Classification Dataset (Diverse queries across 6 core intents)
2. Synthetic User Behavior Profiles for K-Means Clustering
3. Synthetic Prompt-Response-Quality Samples for Quality Regressor
"""

INTENT_DATA = [
    # 1. Technical / Coding
    ("How do I fix TypeError: cannot read property of undefined in JavaScript?", "technical_coding"),
    ("Write a Python script to scrape HTML tables using BeautifulSoup", "technical_coding"),
    ("How does binary search work and what is its time complexity?", "technical_coding"),
    ("Debug this React useEffect infinite loop code", "technical_coding"),
    ("Explain the difference between SQL and NoSQL databases with code examples", "technical_coding"),
    ("How do I configure Docker container with multi-stage build?", "technical_coding"),
    ("Can you implement a Dijkstra shortest path algorithm in C++?", "technical_coding"),
    ("What causes a memory leak in Node.js event listeners?", "technical_coding"),
    ("Write a REST API endpoint in Express.js for user authentication with JWT", "technical_coding"),
    ("How do I invert a binary tree in Python recursively?", "technical_coding"),
    ("Explain async/await under the hood with Promise queue microtasks", "technical_coding"),
    ("Help me optimize this slow MongoDB aggregation query", "technical_coding"),
    ("Write CSS for a responsive grid layout with flexbox fallback", "technical_coding"),
    ("How do I implement rate limiting using Redis in FastAPI?", "technical_coding"),
    ("Show me how to mock an external API in Jest unit tests", "technical_coding"),
    ("What is the difference between shallow copy and deep copy in Python?", "technical_coding"),
    ("Create a TypeScript interface for a nested user profile object", "technical_coding"),
    ("How to set up CI/CD pipeline using GitHub Actions for Node app?", "technical_coding"),
    ("Why is my git rebase having merge conflicts on every commit?", "technical_coding"),
    ("Implement a debounce and throttle utility function in vanilla JS", "technical_coding"),
    ("Write a function to validate email addresses using regex in JavaScript", "technical_coding"),
    ("Explain red-black tree rotation operations with code", "technical_coding"),
    ("How do I implement pagination in Mongoose and Express?", "technical_coding"),
    ("Convert this callback hell into Promise chain and async await", "technical_coding"),
    ("Write a recursive function to calculate the nth Fibonacci number in Go", "technical_coding"),

    # 2. Conceptual / Educational
    ("Explain quantum computing in simple terms for a beginner", "conceptual_educational"),
    ("What is the theory of general relativity and how does gravity curve spacetime?", "conceptual_educational"),
    ("Can you explain how transformer neural networks and self-attention work?", "conceptual_educational"),
    ("What is inflation in macroeconomics and what causes stagflation?", "conceptual_educational"),
    ("Explain the difference between mitosis and meiosis in biology", "conceptual_educational"),
    ("How does the human immune system remember previous pathogens?", "conceptual_educational"),
    ("What is the Doppler effect and why does siren pitch change as it passes?", "conceptual_educational"),
    ("Explain the philosophical concept of Plato's allegory of the cave", "conceptual_educational"),
    ("How does photosynthesis convert sunlight into chemical energy?", "conceptual_educational"),
    ("What is the difference between deductive and inductive reasoning?", "conceptual_educational"),
    ("Explain how TCP/IP handshake and packet routing work across the internet", "conceptual_educational"),
    ("What is entropy in thermodynamics and why does disorder always increase?", "conceptual_educational"),
    ("Explain the architectural difference between Monolithic and Microservices", "conceptual_educational"),
    ("How do black holes form and what happens at the event horizon?", "conceptual_educational"),
    ("Can you break down how CRISPR gene editing functions?", "conceptual_educational"),
    ("What is the difference between supervised, unsupervised, and reinforcement learning?", "conceptual_educational"),
    ("Explain how the central limit theorem works in statistics", "conceptual_educational"),
    ("What is string theory in theoretical physics?", "conceptual_educational"),
    ("Teach me the fundamentals of Game Theory and Nash Equilibrium", "conceptual_educational"),
    ("What is dark matter and how do astrophysicists detect its presence?", "conceptual_educational"),

    # 3. Conversational / Casual
    ("Hello! How are you doing today?", "conversational_casual"),
    ("Hey there, what's up?", "conversational_casual"),
    ("Good morning LAMA AI! Hope you're having a great day.", "conversational_casual"),
    ("Tell me a funny joke to make me smile", "conversational_casual"),
    ("Who created you and what can you do?", "conversational_casual"),
    ("I'm feeling a bit tired and overwhelmed today", "conversational_casual"),
    ("Thanks for your help earlier, you're awesome!", "conversational_casual"),
    ("What is your favorite book or movie?", "conversational_casual"),
    ("Hi! Just stopping by to test the chatbot.", "conversational_casual"),
    ("Do you dream or have thoughts when nobody is chatting with you?", "conversational_casual"),
    ("Good afternoon! Can we just chat about random things?", "conversational_casual"),
    ("How's the weather in AI land?", "conversational_casual"),
    ("Goodbye, see you tomorrow!", "conversational_casual"),
    ("Nice to meet you LAMA!", "conversational_casual"),
    ("Hey buddy, how is life treating you?", "conversational_casual"),
    ("Just checking in, have a wonderful evening!", "conversational_casual"),
    ("You are hilarious, thank you so much!", "conversational_casual"),
    ("What are your hobbies if you had any?", "conversational_casual"),
    ("Hello world! Nice chatting with you.", "conversational_casual"),
    ("Are you happy being an artificial intelligence?", "conversational_casual"),

    # 4. Problem Solving / Advice
    ("How should I prepare for a senior software engineer system design interview?", "problem_solving_advice"),
    ("I have trouble waking up early in the morning. What actionable tips can fix this?", "problem_solving_advice"),
    ("How do I negotiate a higher salary offer without sounding greedy?", "problem_solving_advice"),
    ("My laptop fan is running at 100% and battery drains fast. How do I troubleshoot?", "problem_solving_advice"),
    ("How should I prioritize tasks when working on three urgent client deadlines?", "problem_solving_advice"),
    ("I'm experiencing burnout at work. What strategies should I follow to recover?", "problem_solving_advice"),
    ("What is the best way to invest my first $5,000 for long-term compound growth?", "problem_solving_advice"),
    ("How can I improve my public speaking and overcome presentation anxiety?", "problem_solving_advice"),
    ("Should I choose React or Vue for a new fast-paced MVP startup project?", "problem_solving_advice"),
    ("How do I structure a cold outreach email to a hiring manager on LinkedIn?", "problem_solving_advice"),
    ("My WiFi keeps disconnecting every 30 minutes on Windows. How can I resolve it?", "problem_solving_advice"),
    ("What is the best study schedule to learn Python in 30 days?", "problem_solving_advice"),
    ("How do I deal with a difficult coworker who ignores deadlines?", "problem_solving_advice"),
    ("What steps should I take to start a freelance consulting business?", "problem_solving_advice"),
    ("How can I organize my daily schedule using time-blocking techniques?", "problem_solving_advice"),
    ("How to fix bad posture while sitting at a desk all day?", "problem_solving_advice"),
    ("My car makes a squeaking sound when braking. What could be the issue?", "problem_solving_advice"),
    ("How can I save money on groceries without sacrificing nutrition?", "problem_solving_advice"),

    # 5. Creative / Brainstorming
    ("Brainstorm 10 catchy brand names for an eco-friendly coffee startup", "creative_brainstorming"),
    ("Write an opening scene for a cyberpunk sci-fi novel set in Neo-Tokyo", "creative_brainstorming"),
    ("Give me five creative plot twists for a murder mystery screenplay", "creative_brainstorming"),
    ("Help me draft a compelling tagline and elevator pitch for an AI assistant", "creative_brainstorming"),
    ("Write a poetic description of autumn leaves falling in a silent forest", "creative_brainstorming"),
    ("Brainstorm innovative mobile app concepts combining fitness and gamification", "creative_brainstorming"),
    ("Can you compose a short motivational speech for a graduating class?", "creative_brainstorming"),
    ("Give me ideas for a viral social media marketing campaign for a sneaker brand", "creative_brainstorming"),
    ("Write a dialogue between a medieval knight and a time traveler from 2099", "creative_brainstorming"),
    ("Invent a fictional fantasy spell system based on musical frequencies", "creative_brainstorming"),
    ("Write a haiku about the rain falling on a quiet mountain", "creative_brainstorming"),
    ("Brainstorm 7 video game mechanics for a zero-gravity puzzle platformer", "creative_brainstorming"),
    ("Write a short horror story in under two hundred words", "creative_brainstorming"),
    ("Give me catchy YouTube video titles for a tech review channel", "creative_brainstorming"),
    ("Design a fictional superhero whose power revolves around manipulating probabilities", "creative_brainstorming"),
    ("Draft a dramatic monologue for an astronaut stranded on Mars", "creative_brainstorming"),

    # 6. Factual / Direct Inquiry
    ("What is the capital city of Australia?", "factual_inquiry"),
    ("Who wrote the novel 'Pride and Prejudice'?", "factual_inquiry"),
    ("What is the speed of light in vacuum in meters per second?", "factual_inquiry"),
    ("When did the Apollo 11 moon landing take place?", "factual_inquiry"),
    ("What is the boiling point of water at sea level in Celsius and Fahrenheit?", "factual_inquiry"),
    ("What are the primary colors in the RGB color model?", "factual_inquiry"),
    ("Who was the first president of the United States?", "factual_inquiry"),
    ("What is the chemical formula for glucose?", "factual_inquiry"),
    ("How many continents are there on Earth and what are their names?", "factual_inquiry"),
    ("What is the highest mountain peak in the world?", "factual_inquiry"),
    ("When was the United Nations founded?", "factual_inquiry"),
    ("What is the atomic number of Gold in the periodic table?", "factual_inquiry"),
    ("Who painted the Mona Lisa and in which museum is it located?", "factual_inquiry"),
    ("What is the currency of Japan?", "factual_inquiry"),
    ("How far is the Moon from the Earth in kilometers?", "factual_inquiry"),
    ("Who discovered penicillin in 1928?", "factual_inquiry"),
    ("What is the largest ocean on Earth?", "factual_inquiry"),
    ("What year did World War 2 end?", "factual_inquiry"),
]

# Synthetic user history profiles for K-Means clustering (4 clusters)
# Features: [avg_prompt_len, code_query_ratio, curiosity_depth, sentiment_tone, avg_feedback, conciseness_pref]
# Cluster 0: The Pragmatist (Concise, direct, high conciseness pref, short prompts)
# Cluster 1: The Deep-Diver (Technical, high code ratio, long prompts, detailed)
# Cluster 2: The Conversationalist (Warm, friendly, casual, positive sentiment, balanced)
# Cluster 3: The Brainstormer (Creative, exploratory, high curiosity, medium length)
SYNTHETIC_USER_PROFILES = [
    # Pragmatists
    [25.0, 0.2, 0.3, 0.5, 4.2, 0.9],
    [32.0, 0.3, 0.2, 0.6, 4.0, 0.85],
    [18.0, 0.1, 0.2, 0.4, 3.8, 0.95],
    [28.0, 0.25, 0.35, 0.5, 4.1, 0.8],
    [22.0, 0.15, 0.25, 0.45, 3.9, 0.9],
    
    # Deep-Divers
    [120.0, 0.85, 0.8, 0.5, 4.5, 0.15],
    [145.0, 0.9, 0.85, 0.55, 4.7, 0.1],
    [98.0, 0.75, 0.7, 0.6, 4.3, 0.2],
    [130.0, 0.8, 0.9, 0.48, 4.6, 0.12],
    [110.0, 0.82, 0.78, 0.52, 4.4, 0.18],

    # Conversationalists
    [45.0, 0.05, 0.4, 0.85, 4.3, 0.5],
    [50.0, 0.08, 0.45, 0.9, 4.6, 0.45],
    [38.0, 0.02, 0.35, 0.8, 4.1, 0.55],
    [55.0, 0.1, 0.5, 0.88, 4.5, 0.4],
    [42.0, 0.04, 0.38, 0.82, 4.2, 0.52],

    # Brainstormers
    [75.0, 0.15, 0.92, 0.7, 4.4, 0.3],
    [82.0, 0.2, 0.88, 0.75, 4.6, 0.25],
    [68.0, 0.1, 0.85, 0.68, 4.2, 0.35],
    [90.0, 0.25, 0.95, 0.72, 4.5, 0.2],
    [78.0, 0.18, 0.9, 0.74, 4.3, 0.28],
]

CLUSTER_NAMES = {
    0: "The Pragmatist",
    1: "The Deep-Diver",
    2: "The Conversationalist",
    3: "The Brainstormer"
}
