import os

# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingServer.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

# Your application specific imports
from task.models import User, Question


def main():

    users = User.objects.all().order_by('id')

    for user in users:

        user_id = user.id

        print(f"User {user_id}")
        print("*" * 4)

        que = Question.objects.filter(user_id=user_id).order_by('t')

        for q in que:

            t = q.t
            reaction_time = int((q.time_reply - q.time_display).total_seconds()*10**3)
            success = q.reply == q.correct_answer

            to_print = f't:{t}, question: {q.question}, reply: {q.reply}, correct answer: {q.correct_answer}, ' \
                f'success: {"Yes" if success else "No"}, reaction time: {reaction_time} ms'
            print(to_print)

        print()


if __name__ == "__main__":

    main()
