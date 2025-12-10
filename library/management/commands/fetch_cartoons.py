import requests
import time
from django.core.management.base import BaseCommand
from library.models import Cartoon

class Command(BaseCommand):
    help = "Fetch animated cartoon shows from TVMaze API"

    def handle(self, *args, **kwargs):
        page = 0
        total_saved = 0

        while True:
            url = f"https://api.tvmaze.com/shows?page={page}"

            try:
                response = requests.get(url, timeout=10)
            except Exception:
                self.stdout.write(self.style.WARNING(f"Retrying page {page}..."))
                time.sleep(2)
                continue

            if response.status_code != 200:
                break

            shows = response.json()
            if not shows:
                break

            for show in shows:
                genres = show.get("genres", [])

                # Skip shows with no genres
                if not genres:
                    continue

                # Genre detection
                genre_text = " ".join(genres).lower()
                allowed = ["animation", "anime", "kids", "children", "cartoon"]

                if not any(g in genre_text for g in allowed):
                    continue

                # SAFE IMAGE HANDLING
                image_data = show.get("image") or {}

                Cartoon.objects.update_or_create(
                    tvmaze_id=show["id"],
                    defaults={
                        "name": show.get("name"),
                        "summary": show.get("summary"),
                        "genres": ", ".join(genres),
                        "rating": show.get("rating", {}).get("average"),
                        "premiered": show.get("premiered"),

                        # SAFE IMAGE EXTRACTION
                        "image_medium": image_data.get("medium"),
                        "image_original": image_data.get("original"),

                        "status": show.get("status"),
                        "language": show.get("language"),
                        "runtime": show.get("runtime"),
                        "official_site": show.get("officialSite"),
                        "updated": show.get("updated"),
                        "popularity": show.get("weight"),
                    }
                )
                total_saved += 1

            page += 1
            time.sleep(1)

        self.stdout.write(self.style.SUCCESS(f"Saved {total_saved} cartoon shows successfully"))
