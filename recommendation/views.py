from django.shortcuts import redirect, get_object_or_404, render
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from library.models import Cartoon
from .recommender import hybrid_filtering, collaborative_filtering, content_based_filtering, hybrid_with_clustering
from .models import UserRating

@login_required
def update_user_status(request, cartoon_id, status):

    allowed_status = [
        "watchlist",
        "watching",
        "completed",
        "canceled",
        "not_interested",
    ]

    # Validate status
    if status not in allowed_status:
        messages.error(request, "Invalid category selected.")
        return redirect("cartoon_detail", cartoon_id=cartoon_id)

    cartoon = get_object_or_404(Cartoon, tvmaze_id=cartoon_id)

    # Save or update
    user_rating, created = UserRating.objects.update_or_create(
        user=request.user,
        cartoon=cartoon,
        defaults={"status": status},
    )

    # Success message
    msg = f"Added to '{status.replace('_', ' ').title()}'"
    messages.success(request, msg)

    return redirect("cartoon_detail", cartoon_id=cartoon_id)


@login_required
def user_watchlist(request):
    watchlist = UserRating.objects.filter(user=request.user, status='watchlist')
    watching = UserRating.objects.filter(user=request.user, status='watching')
    completed = UserRating.objects.filter(user=request.user, status='completed')
    canceled = UserRating.objects.filter(user=request.user, status='canceled')
    not_interested = UserRating.objects.filter(user=request.user, status='not_interested')

    context = {
        "watchlist": watchlist,
        "watching": watching,
        "completed": completed,
        "canceled": canceled,
        "not_interested": not_interested,
    }
    return render(request, "watchlist.html", context)

@login_required
def recommendations_page(request):

    user_id = request.user.id

    try:
        # ----- HYBRID -----
        hybrid_results = hybrid_filtering(user_id, 20)
        hybrid_ids = {c.id for c in hybrid_results}

        # ----- COLLABORATIVE -----
        collab_all = collaborative_filtering(user_id, 30)
        collab_results = [c for c in collab_all if c.id not in hybrid_ids][:20]
        collab_ids = {c.id for c in collab_results}

        # ----- CONTENT-BASED -----
        content_all, _, _ = content_based_filtering(user_id, 30)
        content_results = [
            c for c in content_all
            if c.id not in hybrid_ids and c.id not in collab_ids
        ][:20]

        content_ids = {c.id for c in content_results}

        # ----- CLUSTER-BASED RECOMMENDATIONS -----
        cluster_results = hybrid_with_clustering(user_id, 15)

        is_personalized = UserRating.objects.filter(user=request.user).exists()

    except Exception as e:
        print("Recommendation Error:", e)

        hybrid_results = []
        collab_results = []
        content_results = []
        cluster_results = []
        is_personalized = False

    return render(request, "recommendations.html", {
        "hybrid_results": hybrid_results,
        "collab_results": collab_results,
        "content_results": content_results,
        "cluster_results": cluster_results,
        "is_personalized": is_personalized,
    })
