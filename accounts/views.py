from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from library.models import Cartoon

User = get_user_model()

# LOGIN
def login_view(request):
    if request.user.is_authenticated:
        return redirect('index')

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        remember_me = request.POST.get('remember_me')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)

            # Session expiry (if remember me unchecked)
            if not remember_me:
                request.session.set_expiry(0)

            messages.success(request, f"Welcome back, {user.username}!")
            return redirect('index')

        messages.error(request, "Invalid username or password.")
        return redirect('login')

    return render(request, 'login.html')


# SIGNUP
def signup_view(request):
    if request.user.is_authenticated:
        return redirect('index')

    if request.method == 'POST':
        # Fields
        first_name = request.POST.get('first_name', '').strip()
        last_name = request.POST.get('last_name', '').strip()
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        phone = request.POST.get('phone', '').strip()
        address = request.POST.get('address', '').strip()
        password = request.POST.get('password', '')
        confirm_password = request.POST.get('confirm_password', '')
        profile_picture = request.FILES.get('profile_picture')

        # Validation
        if not all([first_name, last_name, username, email, phone, password, confirm_password]):
            messages.error(request, "Please fill all required fields.")
            return redirect('signup')

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('signup')

        if User.objects.filter(username=username).exists():
            messages.error(request, "This username already exists.")
            return redirect('signup')

        if User.objects.filter(email=email).exists():
            messages.error(request, "This email is already registered.")
            return redirect('signup')

        if User.objects.filter(phone=phone).exists():
            messages.error(request, "This phone number is already registered.")
            return redirect('signup')

        # Create User
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            address=address,
        )

        if profile_picture:
            user.profile_picture = profile_picture
            user.save()

        messages.success(request, "Account created successfully! Please login.")
        return redirect('login')

    return render(request, 'signup.html')


# LOGOUT
@login_required
def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('login')


def index(request):

    # Your 5 featured show names
    featured_names = [
        "Pokémon",
        "Phineas and Ferb",
        "Shaun the Sheep",
        "Dragon Ball Z",
        "Legion of Super Heroes",
    ]

    # Safety filter — remove adult content
    adult_regex = r"adult|hentai|erotic|mature|18\+"

    featured_shows = Cartoon.objects.filter(
        name__in=featured_names
    ).exclude(
        genres__iregex=adult_regex
    )

    # Other sections
    top_rated = Cartoon.objects.exclude(rating__isnull=True)\
                               .exclude(genres__iregex=adult_regex)\
                               .order_by('-rating')[:12]

    latest = Cartoon.objects.exclude(premiered="")\
                            .exclude(genres__iregex=adult_regex)\
                            .order_by('-premiered')[:12]

    popular = Cartoon.objects.exclude(genres__iregex=adult_regex)\
                             .order_by('-popularity')[:12]

    context = {
        "featured_shows": featured_shows,
        "top_rated_Cartoon": top_rated,
        "recently_premiered": latest,
        "popular_Cartoon": popular,
    }

    return render(request, "index.html", context)

# PROFILE
@login_required
def profile_view(request):
    return render(request, 'profile.html', {'user': request.user})

def about_page(request):
    return render(request, "about.html")
