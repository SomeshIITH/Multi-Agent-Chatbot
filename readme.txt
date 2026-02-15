### Instructions to run the code


1. Extract the Zip file 

2. Because I used langgraph,hugggingface and many other libraries there dependency is in requirements.txt ,just run ->
        pip install -r requirements.txt  

3. If want to see all in one Agent run -> 
        python -m streamlit run mainfrontend.py 

4. If want to use each one of them independently
    for simple Chatbot run ->   
        python -m streamlit run simpleChatbotfrontend.py

    for rag bot run ->
        python -m streamlit run ragChatbotfrontend.py

    for blog/research agent run ->
        python -m streamlit run blogChatbotfrontend.py

5. Some things to consider while prompting to Agent:
    -   Since I use free API of Llama,qwen via GroqCloud ,there is daily limit and rate limiting ,so sometimes 
        output may be not as expected due to limit reached for these API.
        This issue can be resolved by Paid API easily , but then sharing of these API can cause too much Bill.
    
    -   In RAG model , it chunks whole PDF which take time  depending on size of PDF file , if less size/
        page , less time it will take to index them

6. Demo of images of how it works and output is coming in readme.md.
    Can be read via  https://github.com/SomeshIITH/Multi-Agent-Chatbot

