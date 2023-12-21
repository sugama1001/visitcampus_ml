import unittest
import json
from main_API_SOC import (app)

class FlaskAppTest(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # test home
    def test_home_endpoint(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"The API WORK", response.data)

    # test untuk endpoint
    def test_predict_endpoint(self):
        input_data = {
            "EI_text": "saya suka menyendiri",
            "SN_text": "saya selalu mengikuti kata hati",
            "TF_text": "saya suka analisis",
            "JP_text": "bertindak sesuka hati adalah jalan ninjaku"
        }
        response = self.app.post('/predict', json=input_data)

        self.assertEqual(response.status_code, 200)

        # memastikan bahwa data yang diberikan berupa JSON
        json_response = json.loads(response.data)
        self.assertIsInstance(json_response, dict)  # Mengubah assert ke objek dictionary

        # pastikan hasil prediksi sesuai hasil yang diharapkan
        for value in json_response.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    # test jika inputan berupa angka
    def test_predict_endpoint_with_numbers(self):
        input_data = {
            "EI_text": 123,  # contoh input angka
            "SN_text": "saya selalu mengikuti kata hati",
            "TF_text": "saya suka analisis",
            "JP_text": "nah kamu babi"
        }
        response = self.app.post('/predict', json=input_data)

        self.assertEqual(response.status_code, 400)  # Expecting Bad Request

    # test jika inputan kosong
    def test_predict_endpoint_with_empty_text(self):
        input_data = {
            "EI_text": "",  # teks kosong
            "SN_text": "saya selalu mengikuti kata hati",
            "TF_text": "saya suka analisis",
            "JP_text": "nah kamu babi"
        }
        response = self.app.post('/predict', json=input_data)

        self.assertEqual(response.status_code, 400)  # Expecting Bad Request
        json_response = json.loads(response.data)
        self.assertIn("error", json_response)
        self.assertIn(f"Teks input tidak boleh kosong", json_response["error"])


if __name__ == '__main__':
    unittest.main()
