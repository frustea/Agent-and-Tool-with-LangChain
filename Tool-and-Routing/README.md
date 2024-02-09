## How to Build Tool and Routing with LangChain

Here is a  codes to build an essential pipline for defining the consitent function and feed it to the LLM model, using the LangChain tool model.
In this method we define the ``` arga_schema``` and set it to the desired schema of our function. 
Here is an example: 
```
class Yourfunc(BaseModel):
    "description of function  "
    your_first_paramter: float =Field(description="description of your first parameter ")
   

@tool(args_schema=Triangle)
def check_triangle(your_first_paramter:float)-> dict:
    "description of your function"
    ## main body of function 

   return ## your desired output 
```

In the next step, to feed multiple functions and ask LLM to route between them we can define routing function like the following:
```
def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "your first function": func1, 
            "your second function": func2,
        }
        return tools[result.tool].run(result.tool_input)
    
chain = prompt | model | OpenAIFunctionsAgentOutputParser() | route
```
