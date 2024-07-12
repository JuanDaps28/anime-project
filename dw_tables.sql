drop table if exists animes_ratings;
drop table if exists animes_catalog;

create table animes_catalog(
	anime_id int not null,
	name varchar(150) not null,
	genre varchar(188) not null,
	type varchar(11) not null,
	episodes smallint null,
	synopsis text not null,
	deleted boolean not null,
	
	primary key(anime_id)
);

create table animes_ratings(
	rating_id int generated always as identity,
	user_id int not null,
	anime_id int not null,
	rating int not null,
	
	primary key(rating_id),
	foreign key(anime_id)
	references animes_catalog(anime_id)
);
