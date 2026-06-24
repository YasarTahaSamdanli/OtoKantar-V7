FROM php:8.3-apache

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libpq-dev libzip-dev unzip zip \
    && docker-php-ext-install pdo_mysql pdo_pgsql zip bcmath \
    && a2enmod rewrite \
    && rm -rf /var/lib/apt/lists/*

COPY --from=composer:2 /usr/bin/composer /usr/bin/composer

ENV APACHE_DOCUMENT_ROOT=/var/www/html/public

RUN sed -ri "s!/var/www/html!${APACHE_DOCUMENT_ROOT}!g" /etc/apache2/sites-available/*.conf \
    && sed -ri "s!/var/www/!${APACHE_DOCUMENT_ROOT}!g" /etc/apache2/apache2.conf /etc/apache2/conf-available/*.conf

WORKDIR /var/www/html

COPY docker/dev-entrypoint.sh /usr/local/bin/otokantar-dev-entrypoint

RUN chmod +x /usr/local/bin/otokantar-dev-entrypoint

EXPOSE 80

ENTRYPOINT ["otokantar-dev-entrypoint"]
CMD ["apache2-foreground"]
