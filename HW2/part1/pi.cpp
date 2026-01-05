#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>

#ifdef __AVX2__
#include <immintrin.h>   
#endif

typedef struct {
    __m256i state;
} rng8_t;

pthread_mutex_t mutex_sum;
int total_hit = 0;

static inline uint64_t splitmix64(uint64_t *x){
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline rng8_t rng8_init(uint64_t seed){
    uint32_t seeds[8];
    for (int i = 0; i < 8; i++) {
        uint64_t v = splitmix64(&seed);
        uint32_t s = (uint32_t)(v ^ (v >> 32));
        if (s == 0) s = 0x6D25352B ^ i;   
        seeds[i] = s;
    }
    rng8_t r;
    r.state = _mm256_loadu_si256((__m256i*)seeds);
    return r;
}

static inline __m256i xorshift32_vec(__m256i *s){
    __m256i x = *s;
    x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
    x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
    *s = x;
    return x;
}

static inline __m256 rng8_next_signed(rng8_t *r){
    __m256i raw = xorshift32_vec(&r->state);

    const __m256 add   = _mm256_set1_ps(2147483648.0f);        // 2^31
    const __m256 scale = _mm256_set1_ps(1.0f/4294967296.0f);   // 2^-32
    __m256 f = _mm256_cvtepi32_ps(raw);
    f = _mm256_add_ps(f, add);
    f = _mm256_mul_ps(f, scale);

    f = _mm256_fmsub_ps(f, _mm256_set1_ps(2.0f), _mm256_set1_ps(1.0f));

    return f;
}


void *calc_pi(void *local_toss){
	int local_toss_i = *(int *)local_toss;
	int local_hit = 0;
	int i = 0;
#ifdef __AVX2__
	const __m256 one = _mm256_set1_ps(1.0);
	rng8_t rng = rng8_init(123456);
	for(i = 0; i < local_toss_i; i += 8){
		__m256 xv = rng8_next_signed(&rng);
		__m256 yv = rng8_next_signed(&rng);


		__m256 dist =
	    	#ifdef __FMA__
			_mm256_fmadd_ps(xv, xv, _mm256_mul_ps(yv, yv));
	    	#else
			_mm256_add_ps(_mm256_mul_ps(xv, xv), _mm256_mul_ps(yv, yv));
	    	#endif

		__m256 cmp = _mm256_cmp_ps(dist, one, _CMP_LE_OQ);
		int mask = _mm256_movemask_ps(cmp);
		local_hit += __builtin_popcount((unsigned)mask);

	}
#else
    for (; i < local_toss_i; ++i) {
        double x = 2.0 * (double)rand_r(&seed) / RAND_MAX - 1.0;
        double y = 2.0 * (double)rand_r(&seed) / RAND_MAX - 1.0;
        double dist = x * x + y * y;
        if (dist <= 1.0) local_hit++;
    }
#endif
	pthread_mutex_lock(&mutex_sum);
	total_hit += local_hit;
	pthread_mutex_unlock(&mutex_sum);

	return NULL;
}

int main(int argc, char * argv[]){
	int thd_cnt;
	long long int toss;
	if (argc != 3) {
		printf("Usage: ./pi.out {CPU core} {Number of tosses}\n");
		return 1;
	}
	thd_cnt = atoi(argv[1]);
	toss = atoll(argv[2]);

	int local_toss = toss / thd_cnt;
	int rem = toss % thd_cnt;
	int first_toss = local_toss + rem;


	pthread_t threads[thd_cnt];

	pthread_mutex_init(&mutex_sum, NULL);
	for(int i = 0; i < thd_cnt; ++i){
		if(i == 0) pthread_create(threads + i, NULL, &calc_pi, (void *)&first_toss);
		else pthread_create(threads + i, NULL, &calc_pi, (void *)&local_toss);
	}

	for(int i = 0; i < thd_cnt; ++i){
		pthread_join(threads[i], NULL);
	}

	double pi = (double)total_hit / (double)toss * 4.0;
	pthread_mutex_destroy(&mutex_sum);

	printf("%f\n", pi);
	return 0;
}
