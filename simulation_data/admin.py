from django.contrib import admin

from . models import Simulation, Post


class SimulationAdmin(admin.ModelAdmin):
    list_display = (
        "id", "n_session", "n_iteration_per_session",
        "n_iteration_between_session", "grid_size", "teacher_model",
        "learner_model", "param_labels", "param_values",
        "param_lower_bounds", "param_upper_bounds",
        "seed"
    )


class PostAdmin(admin.ModelAdmin):
    list_display = (
        "id", "simulation", "param_label", ) # "mean", "std")


admin.site.register(Simulation, SimulationAdmin)
admin.site.register(Post, PostAdmin)