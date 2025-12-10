from django.shortcuts import render, get_object_or_404
from django.core.paginator import Paginator
from urllib3 import request
from recommendation.models import UserRating
from .models import Cartoon
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.db.models import Avg, Count
import json

# LIST VIEW - All Cartoons
def cartoon_list(request):
    query = request.GET.get("q", "")

    cartoons = Cartoon.objects.all().order_by('-popularity')

    # Remove adult content
    cartoons = cartoons.exclude(genres__iregex="adult|hentai|erotic|mature|18\\+")

    if query:
        cartoons = cartoons.filter(name__icontains=query)

    paginator = Paginator(cartoons, 24)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    total = paginator.num_pages
    current = page_obj.number

    # Default values to avoid UnboundLocalError
    start = 1
    end = total

    # Build smart pagination window (max 8 items)
    if total > 8:
        start = max(current - 3, 1)
        end = min(current + 3, total)

    page_range = range(start, end + 1)

    # Flags for ellipsis conditions
    show_first = start > 1
    show_last = end < total

    context = {
        "page_obj": page_obj,
        "page_range": page_range,
        "show_first": show_first,
        "show_last": show_last,
        "total_pages": total,
        "query": query,
    }

    return render(request, "cartoon_list.html", context)

# DETAIL VIEW
def cartoon_detail(request, cartoon_id):
    if not request.user.is_authenticated:
        return render(request, "login_required.html")

    cartoon = get_object_or_404(Cartoon, tvmaze_id=cartoon_id)
    genres = cartoon.genres.split(",") if cartoon.genres else []

    # Adult content protection
    adult_keywords = ["adult", "hentai", "erotic", "mature", "18+"]
    for g in genres:
        if any(a in g.lower() for a in adult_keywords):
            return render(request, "not_allowed.html")

    # USER RATING + REVIEW
    user_rating_value = None
    user_review_text = ""

    if request.user.is_authenticated:
        ur = UserRating.objects.filter(user=request.user, cartoon=cartoon).first()
        if ur:
            user_rating_value = ur.rating or 0
            user_review_text = ur.review or ""

    # ALL REVIEWS FOR DISPLAY
    reviews = UserRating.objects.filter(
        cartoon=cartoon
    ).exclude(review__isnull=True).exclude(review="").order_by("-created_at")

    # AVERAGE RATING + COUNT
    stats = UserRating.objects.filter(cartoon=cartoon).aggregate(
        avg=Avg("rating"),
        total=Count("rating")
    )

    avg_rating = round(stats["avg"], 1) if stats["avg"] else 0
    total_ratings = stats["total"]

    context = {
        "cartoon": cartoon,
        "genres": genres,
        # rating & review
        "user_rating_value": user_rating_value,
        "user_review_text": user_review_text,
        "rating_range": range(1, 10 + 1),
        # for public display
        "reviews": reviews,
        "avg_rating": avg_rating,
        "total_ratings": total_ratings,
    }

    return render(request, "cartoon_detail.html", context)

# SAVE RATING (Ajax)
@login_required
def rate_cartoon(request, cartoon_id):
    cartoon = get_object_or_404(Cartoon, tvmaze_id=cartoon_id)
    data = json.loads(request.body)
    rating_value = int(data.get("rating", 0))

    UserRating.objects.update_or_create(
        user=request.user,
        cartoon=cartoon,
        defaults={"rating": rating_value}
    )

    return JsonResponse({"status": "success", "rating": rating_value})

# SAVE REVIEW (Ajax)
@login_required
def review_cartoon(request, cartoon_id):
    cartoon = get_object_or_404(Cartoon, tvmaze_id=cartoon_id)
    data = json.loads(request.body)
    review_text = data.get("review", "")

    UserRating.objects.update_or_create(
        user=request.user,
        cartoon=cartoon,
        defaults={"review": review_text}
    )

    return JsonResponse({"status": "success", "review": review_text})


# GENRES PAGE
def genre_list(request):
    # Fetch all cartoons except adult content
    cartoons = Cartoon.objects.exclude(genres__isnull=True).exclude(genres="")

    # Adult words to filter out
    adult_keywords = ["adult", "hentai", "erotic", "mature", "18+"]

    genre_set = set()

    for c in cartoons:
        for g in c.genres.split(","):
            g_clean = g.strip()

            # Skip adult genres
            if any(a in g_clean.lower() for a in adult_keywords):
                continue

            genre_set.add(g_clean)

    genres = sorted(list(genre_set))

    context = {
        "genres": genres
    }
    return render(request, "genre_list.html", context)

# GENRE DETAIL PAGE
def genre_detail(request, genre_name):

    # Reject adult genre entirely
    adult_keywords = ["adult", "hentai", "erotic", "mature", "18+"]

    if any(a in genre_name.lower() for a in adult_keywords):
        return render(request, "not_allowed.html")

    # Get cartoons containing this genre (ignoring adult ones)
    cartoons = Cartoon.objects.filter(
        genres__icontains=genre_name
    )

    # Remove adult content from results
    filtered_cartoons = []
    for c in cartoons:
        if not any(a in c.genres.lower() for a in adult_keywords):
            filtered_cartoons.append(c)

    context = {
        "genre_name": genre_name,
        "cartoons": filtered_cartoons
    }
    return render(request, "genre_detail.html", context)