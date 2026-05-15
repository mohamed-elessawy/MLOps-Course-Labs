from locust import HttpUser, between, task


class ChurnAPIUser(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(2)
    def root(self):
        self.client.get("/")

    @task(5)
    def predict(self):
        self.client.post(
            "/predict",
            json={
                "CreditScore": 619,
                "Geography": "France",
                "Gender": "Female",
                "Age": 42,
                "Tenure": 2,
                "Balance": 0.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348.88,
            },
        )
