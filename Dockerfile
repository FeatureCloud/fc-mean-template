FROM python:3.8-slim

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y supervisor nginx
RUN pip3 install --upgrade pip

COPY requirements.txt /requirements.txt
RUN pip3 install -r ./requirements.txt

COPY server_config/supervisord.conf /supervisord.conf
COPY server_config/nginx /etc/nginx/sites-available/default
COPY server_config/docker-entrypoint.sh /entrypoint.sh

COPY . /app

EXPOSE 9000 9001

ENTRYPOINT ["sh", "/entrypoint.sh"]
