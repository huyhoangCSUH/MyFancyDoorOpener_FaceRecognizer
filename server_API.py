import web
import json
from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return 'Under construction!'

@app.route('/getinfo')
def get_info():
    with open("auth_file.txt", "rb") as fin:
        person_info = fin.read()
        # print person_info
        return person_info

'''
urls = (
    '/getinfo', 'get_info',
    '/setauth', 'set_auth'
)

app = web.application(urls, globals())


class get_info:
    def GET(self):
        with open("auth_file.txt", "rb") as fin:
            person_info = fin.read()
            #print person_info
            return person_info

if __name__ == "__main__":
    app.run()
'''