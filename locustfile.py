from locust import HttpUser, task, between

class ChatbotUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def consulta_chatbot(self):
        self.client.post("/consulta", json={"query": "Dame un resumen del PDF"})
