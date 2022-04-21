#!/bin/sh

if [ "$DATABASE" = "postgres" ]
then
    echo "Waiting for postgres..."

    while ! nc -z $SQL_HOST $SQL_PORT; do
      sleep 0.1
    done

    echo "PostgreSQL started"
fi

#python3 manage.py flush --no-input
python3 manage.py makemigrations
python3 manage.py migrate
#python3 manage.py migrate --database=gdpr_log
python3 manage.py collectstatic --noinput
echo "from django.contrib.auth import get_user_model; User = get_user_model(); User.objects.create_superuser('root', 'user@mail.com', 'user')" | python3 manage.py shell
exec "$@"
