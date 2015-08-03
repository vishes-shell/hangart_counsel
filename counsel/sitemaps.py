from django.contrib import sitemaps
from django.core.urlresolvers import reverse


class StaticCounselSitemap(sitemaps.Sitemap):
    priority = 1
    changefreq = 'monthly'

    def items(self):
        return ['counsel', 'counsel_en']

    def location(self, item):
        return reverse(item)
