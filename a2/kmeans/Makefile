.KEEP_STATE:

CC = gcc

CFLAGS = -Wall -Wextra -Wno-unused -O3

OMPFLAGS = -fopenmp $(CFLAGS)

LDFLAGS =

H_FILES = kmeans.h

COMM_SRC = file_io.c util.c

LOCKS_PREFIX = ./locks
LOCKS_FLAGS = -I$(LOCKS_PREFIX)

# all: kmeans_seq
# all: kmeans_seq kmeans_omp_naive kmeans_omp_reduction
all:  kmeans_seq kmeans_omp_naive_delta kmeans_omp_reduction kmeans_omp_naive kmeans_omp_critical kmeans_omp_nosync_lock kmeans_omp_pthread_mutex_lock kmeans_omp_pthread_spin_lock kmeans_omp_tas_lock kmeans_omp_ttas_lock kmeans_omp_array_lock kmeans_omp_clh_lock

kmeans_seq: main.o file_io.o util.o seq_kmeans.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
kmeans_omp_naive_delta: main.o file_io.o util.o omp_naive_kmeans_delta.o
	$(CC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)
kmeans_omp_reduction: main.o file_io.o util.o omp_reduction_kmeans.o
	$(CC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

kmeans_omp_naive: main.o file_io.o util.o omp_naive_kmeans.o
	$(CC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)
kmeans_omp_critical: main.o file_io.o util.o omp_critical_kmeans.o
	$(CC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

kmeans_omp_nosync_lock: main.o file_io.o util.o omp_lock_kmeans.o $(LOCKS_PREFIX)/nosync_lock.o
	$(CC) $(OMPFLAGS) -pthread $^ -o $@ $(LDFLAGS)
kmeans_omp_pthread_mutex_lock: main.o file_io.o util.o omp_lock_kmeans.o $(LOCKS_PREFIX)/pthread_mutex_lock.o
	$(CC) $(OMPFLAGS) -pthread $^ -o $@ $(LDFLAGS)
kmeans_omp_pthread_spin_lock: main.o file_io.o util.o omp_lock_kmeans.o $(LOCKS_PREFIX)/pthread_spin_lock.o
	$(CC) $(OMPFLAGS) -pthread $^ -o $@ $(LDFLAGS)
kmeans_omp_tas_lock: main.o file_io.o util.o omp_lock_kmeans.o $(LOCKS_PREFIX)/tas_lock.o
	$(CC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)
kmeans_omp_ttas_lock: main.o file_io.o util.o omp_lock_kmeans.o $(LOCKS_PREFIX)/ttas_lock.o
	$(CC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)
kmeans_omp_array_lock: main.o file_io.o util.o omp_lock_kmeans.o $(LOCKS_PREFIX)/array_lock.o
	$(CC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)
kmeans_omp_clh_lock: main.o file_io.o util.o omp_lock_kmeans.o $(LOCKS_PREFIX)/clh_lock.o
	$(CC) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)


main.o: main.c $(H_FILES)
	$(CC) $(CFLAGS) -c $< -o $@

seq_kmeans.o: seq_kmeans.c $(COMM_SRC) $(H_FILES)
	$(CC) $(CFLAGS) -c $< -o $@
omp_naive_kmeans_delta.o: omp_naive_kmeans_delta.c $(COMM_SRC) $(H_FILES)
	$(CC) $(OMPFLAGS) -c $< -o $@
omp_naive_kmeans.o: omp_naive_kmeans.c $(COMM_SRC) $(H_FILES)
	$(CC) $(OMPFLAGS) -c $< -o $@
omp_critical_kmeans.o: omp_critical_kmeans.c $(COMM_SRC) $(H_FILES)
	$(CC) $(OMPFLAGS) -c $< -o $@
omp_reduction_kmeans.o: omp_reduction_kmeans.c $(COMM_SRC) $(H_FILES)
	$(CC) $(OMPFLAGS) -c $< -o $@
omp_lock_kmeans.o: omp_lock_kmeans.c $(COMM_SRC) $(H_FILES) 
	$(CC) $(OMPFLAGS) $(LOCKS_FLAGS) -c $< -o $@


file_io.o: file_io.c
#	$(CC) $(CFLAGS) -c $< -o $@
# Hint : why is OMPFLAGS used here?	(when using it, need to include -fopenmp to LDFLAGS too)
	$(CC) $(OMPFLAGS) -c $< -o $@

util.o: util.c
	$(CC) $(CFLAGS) -c $< -o $@


$(LOCKS_PREFIX)/nosync_lock.o: $(LOCKS_PREFIX)/nosync_lock.c 
	$(CC) $(CFLAGS) -pthread $(LOCKS_FLAGS) -c $< -o $@	
$(LOCKS_PREFIX)/pthread_mutex_lock.o: $(LOCKS_PREFIX)/pthread_mutex_lock.c 
	$(CC) $(CFLAGS) -pthread $(LOCKS_FLAGS) -c $< -o $@	
$(LOCKS_PREFIX)/pthread_spin_lock.o: $(LOCKS_PREFIX)/pthread_spin_lock.c 
	$(CC) $(CFLAGS) -pthread $(LOCKS_FLAGS) -c $< -o $@	
$(LOCKS_PREFIX)/tas_lock.o: $(LOCKS_PREFIX)/tas_lock.c 
	$(CC) $(CFLAGS) $(LOCKS_FLAGS) -c $< -o $@	
$(LOCKS_PREFIX)/ttas_lock.o: $(LOCKS_PREFIX)/ttas_lock.c
	$(CC) $(CFLAGS) $(LOCKS_FLAGS) -c $< -o $@	
$(LOCKS_PREFIX)/array_lock.o: $(LOCKS_PREFIX)/array_lock.c
	$(CC) $(CFLAGS) $(LOCKS_FLAGS) -c $< -o $@	
$(LOCKS_PREFIX)/clh_lock.o: $(LOCKS_PREFIX)/clh_lock.c 
	$(CC) $(CFLAGS) $(LOCKS_FLAGS) -c $< -o $@	


clean:
	rm -rf *.o kmeans_seq kmeans_omp_naive_delta kmeans_omp_reduction kmeans_omp_naive kmeans_omp_critical kmeans_omp_nosync_lock kmeans_omp_pthread_mutex_lock kmeans_omp_pthread_spin_lock kmeans_omp_tas_lock kmeans_omp_ttas_lock kmeans_omp_array_lock kmeans_omp_clh_lock locks/*.o 
