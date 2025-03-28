
#define FIRST_DIM 1 

                                        
#define FIRST_REAL double
#define FIRST_REAL_STRING "double" 

                                        
#define FIRST_MAXDEPTH 32 

                                        
#define N_PARAMS_0(param_field)
#define N_PARAMS_1(param_field) N_PARAMS_0(param_field) param_field[0]
#define N_PARAMS_2(param_field) N_PARAMS_1(param_field), param_field[1]
#define N_PARAMS_3(param_field) N_PARAMS_2(param_field), param_field[2]
#define N_PARAMS_4(param_field) N_PARAMS_3(param_field), param_field[3]
#define N_PARAMS_5(param_field) N_PARAMS_4(param_field), param_field[4]
#define N_PARAMS_6(param_field) N_PARAMS_5(param_field), param_field[5] 

                                        
#define ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAXDEPTHS  \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH(1, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH(1, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH(2, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH(2, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH(3, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH(3, double, 64) 

                                        
#define ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAXDEPTHS_DIMS2_REALS2_MAXDEPTHS2  \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(1, double, 32, 1, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(1, double, 32, 1, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(1, double, 32, 2, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(1, double, 32, 2, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(1, double, 32, 3, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(1, double, 32, 3, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(1, double, 64, 1, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(1, double, 64, 1, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(1, double, 64, 2, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(1, double, 64, 2, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(1, double, 64, 3, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(1, double, 64, 3, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(2, double, 32, 1, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(2, double, 32, 1, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(2, double, 32, 2, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(2, double, 32, 2, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(2, double, 32, 3, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(2, double, 32, 3, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(2, double, 64, 1, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(2, double, 64, 1, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(2, double, 64, 2, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(2, double, 64, 2, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(2, double, 64, 3, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(2, double, 64, 3, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(3, double, 32, 1, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(3, double, 32, 1, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(3, double, 32, 2, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(3, double, 32, 2, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(3, double, 32, 3, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(3, double, 32, 3, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(3, double, 64, 1, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(3, double, 64, 1, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(3, double, 64, 2, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(3, double, 64, 2, double, 64) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(3, double, 64, 3, double, 32) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(3, double, 64, 3, double, 64) 

                                        
#define ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_GRIDS  \
ELSE_IF_COMBINATION_OF_DIM_REAL_GRID(1, double, InnerGrid, 1) \
ELSE_IF_COMBINATION_OF_DIM_REAL_GRID(2, double, InnerGrid, 1) \
ELSE_IF_COMBINATION_OF_DIM_REAL_GRID(3, double, InnerGrid, 1) 

                                        
#define ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAPS  \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAP(1, double, Identity, 0) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAP(2, double, Identity, 0) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAP(2, double, Henon, 3) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAP(2, double, Ikeda, 5) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAP(2, double, StandardBO2, 2) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAP(3, double, Identity, 0) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAP(3, double, Chua, 6) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAP(3, double, Lorenz, 5) 

                                        
#define ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAXDEPTHS_GRIDS_MAPS  \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(1, double, 32, InnerGrid, 1, Identity, 0) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(1, double, 64, InnerGrid, 1, Identity, 0) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(2, double, 32, InnerGrid, 1, Identity, 0) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(2, double, 32, InnerGrid, 1, Henon, 3) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(2, double, 32, InnerGrid, 1, Ikeda, 5) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(2, double, 32, InnerGrid, 1, StandardBO2, 2) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(2, double, 64, InnerGrid, 1, Identity, 0) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(2, double, 64, InnerGrid, 1, Henon, 3) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(2, double, 64, InnerGrid, 1, Ikeda, 5) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(2, double, 64, InnerGrid, 1, StandardBO2, 2) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(3, double, 32, InnerGrid, 1, Identity, 0) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(3, double, 32, InnerGrid, 1, Chua, 6) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(3, double, 32, InnerGrid, 1, Lorenz, 5) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(3, double, 64, InnerGrid, 1, Identity, 0) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(3, double, 64, InnerGrid, 1, Chua, 6) \
ELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(3, double, 64, InnerGrid, 1, Lorenz, 5) 
