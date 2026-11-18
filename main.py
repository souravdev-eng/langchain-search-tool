from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    information = """I started my working life in a supermarket. Software engineering wasn’t the obvious next step—but I became curious about technology, taught myself to code, and gradually turned that curiosity into a professional career.
        That journey shaped how I work today. I’m comfortable entering unfamiliar territory, breaking complicated problems into smaller parts, and learning what’s needed to move forward. I don’t pretend to know everything, but I’m persistent about finding the right answer.
        Today, I’m a FullStack Developer at Astria Digital. Most of my work is frontend-focused, building with React, while I also contribute to backend development using Node.js, AWS Serverless, PostgreSQL, and MongoDB. My work has included microfrontends, reusable components and libraries, backend APIs, code reviews, mentoring, CI/CD, and deployments.
        What interests me most is the point where different parts of a system come together—where a frontend decision affects an API, a data model influences performance, or deployment choices affect how reliably a product can evolve.
        I bring an end-to-end view of product engineering. I can move between user experience, reusable frontend architecture, APIs, databases, and delivery—helping teams make decisions that work across the product, not just within one layer.
        I value code that others can understand and extend, collaboration that improves team standards, and solutions that address the real product problem. Whether I’m building features, reviewing code, mentoring teammates, or supporting a deployment, my goal is to leave both the product and the codebase stronger.
        My direction is to become an engineer who can take increasingly complex product problems from idea to reliable implementation. I’m always happy to connect with engineers, builders, and product teams who care about thoughtful engineering, ownership, and continuous growth.
        """
    summary_template = """
        given the following information about a person, summarize it in a few sentences:
        {information}
        Like:
        1. A short summary
        2. Two interesting facts about the person
        """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(model="gpt-5", temperature=0)

    chain = summary_prompt_template | llm
    chain_result = chain.invoke(input={"information": information})

    print(chain_result.content)


if __name__ == "__main__":
    main()
