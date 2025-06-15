from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.tools import StructuredTool
from dotenv import load_dotenv
import os
import uuid
import sqlite3


class CancerDataAgent:
    def __init__(self, chat_model='gpt-4o-mini'):
        load_dotenv()

        self.db_structure = self.get_db_structure()

        self.tools = [
            StructuredTool.from_function(
                name='run_sql_query',
                func=self.run_sql_query, 
                description=f'''
                            Provide availity to query data from patient database (SQLITE3).
                            Here is a structure of our DB.

                            {self.db_structure}

                            run_sql_query() can:

                            - given SQL query, execute query and get result.

                            input: string (SQL query)
                            output: string (result of the query execution)
                            ''',
            )
        ]
        self.agent = self.initialize_chat_agent(chat_model)
    
    def get_db_structure(self):
        conn = sqlite3.connect(os.getenv("DB_PATH"))
        cursor = conn.cursor()
        db_type = "SQLite3"

        db_description = """
        This is a description of DB tables and columns.
        This DB is used in cancer institute.
        We want you to generate useful insights from this database.
        """

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            columns = cursor.fetchall()
            column_descriptions = [f"{col[1]}({col[2]})" for col in columns]
            column_str = ", ".join(column_descriptions)
            db_description += f"\ntable name: {table_name}\ncolumns: {column_str}\n----------\n"
        
        cursor.close()
        conn.close()


        db_structure = f"""
            database type is {db_type}
            Here is a description of the database.

            database description: {db_description}

            You can use the db_query_and_return_result tool to execute SQL queries on the database and return results.
        """
        return db_structure

    def initialize_chat_agent(self, chat_model):
        # provide context as system message
        db_context = self.get_db_structure()
        system_message = f'''
            You are a knowledgeable assistant that helps a Cancer Institute in Australia NSW.
            You can use database to answer all the question.
            {db_context}
        '''
        memory = MemorySaver()
        llm = ChatOpenAI(
            model=chat_model,
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        agent = create_react_agent(llm, self.tools, prompt=system_message, checkpointer=memory, debug=False)
        return agent

    def start_chat(self):
        self.color_print("####################################################################################\n", "instruction")
        self.color_print("Welcome to the Cancer Data Agent service. I can give you any insights related to Patients and Hospital. \n", "instruction")
        self.color_print("####################################################################################\n", "instruction")
        session_id = str(uuid.uuid4())


        while True:
            user_input = input("ENTER ('q' to quit) : \n")

            if user_input.lower() in ('q', 'quit', 'exit'):
                self.color_print('\nGoodbye!\n\n', 'agent')
                break
            
            print()

            config = {'configurable': {'thread_id': session_id}}
            inputs = {'messages': [('user', user_input)]}

            response = self.agent.invoke(inputs,config)['messages'][-1].content

            self.color_print(f"{response}\n", 'agent')
    
    def run_sql_query(self, query: str) -> str:
        """
            Provide avility to query data from patient database (SQLITE3).
            run_sql_query() can:
            - given SQL query, execute query and get result.

            input: string (SQL query)
            output: string (result of the query execution)
        """
        self.color_print('Tool called - run_sql_query', 'debug')


        conn = sqlite3.connect(os.getenv("DB_PATH"))
        cursor = conn.cursor()

        try:
            cursor.execute(query)
            self.color_print(f'SQL query called = {query}', 'debug')
            result = cursor.fetchall()
            return str(result)
        except Exception as e:
            self.color_print(f'Error during run_sql_query : {str(e)}', 'debug')
            return f"Error: {e}"
        finally:
            cursor.close()
            conn.close()
    
    def color_print(self, message: str, type: str) -> None:
        '''
        Print colored message in terminal
        Available types: 'system', 'agent', 'instruction'
        
        'system' prints in BLUE
        'agent' prints in GREEN
        'instruction' prints in YELLOW
        All other input for types print in WHITE
        '''

        if(type == 'debug' and os.getenv("DEBUG_MODE") == 0):
            return

        match type:
            case 'debug':
                color = '\u001b[33m' # yellow
            case 'system':
                # color = '\u001b[35m ### ' # magenta
                color = '\u001b[36m ###' # blue
            case 'agent':
                color = '\u001b[32m' # green
            case 'instruction':
                color = '\u001b[36m' # blue
            case _:
                color = '\u001b[37m' # white
        
        print(f'{color}{message}\u001b[0m')
        return



if __name__ == '__main__':
    chat_instance = CancerDataAgent()
    chat_instance.start_chat()