from django.contrib import admin

from . models import Item


# Register your models here.
class ItemAdmin(admin.ModelAdmin):
    list_display = (
        "id", "p_recall", "timestamp", "delta",
        "user_id", "learning_language", "ui_language",
        "lexeme_id", "lexeme_string",
        "history_seen", "history_correct",
        "session_seen", "session_correct"
    )


admin.site.register(Item, ItemAdmin)
