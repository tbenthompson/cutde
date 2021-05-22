<%namespace module="cutde.mako_helpers" import="*"/>
<%namespace name="common" file="common.cu"/>

${common.defs()}

WITHIN_KERNEL 
int buffer_alloc(GLOBAL_MEM int* next_ptr, int n_values) {
    printf("alloc! %i %i \n", *next_ptr, n_values);
    % if cluda_backend == 'cuda':
        return atomicAdd(next_ptr, n_values);
    % else:
        return atomic_add(next_ptr, n_values);
    % endif
}

WITHIN_KERNEL
bool in(int target, GLOBAL_MEM int* arr, int n_arr) {
    // Could be faster by keeping arr sorted and doing binary search. 
    // but that is probably premature optimization.
    for (int i = 0; i < n_arr; i++) {
        if (target == arr[i]) {
            return true;
        }
    }
    return false;
}

WITHIN_KERNEL
int new_reference_idx(int idx, int max_idx, GLOBAL_MEM int* prev_arr, int n_prev) {
    while (true) {
        idx = (idx + 3) % max_idx; 
        idx -= idx % 3;
        if (!in(idx, prev_arr, n_prev)) {
            return idx;
        }
    }
}

struct MatrixIndex {
    int row;
    int col;
};

% for matrix_dim in ["rows", "cols"]:
WITHIN_KERNEL struct MatrixIndex argmax_abs_not_in_list_${matrix_dim}(GLOBAL_MEM Real* data, int n_data_rows, int n_data_cols, GLOBAL_MEM int* prev, int n_prev) 
{
    struct MatrixIndex max_idx;
    Real max_val = -1;
    for (int i = 0; i < n_data_rows; i++) {
        for (int j = 0; j < n_data_cols; j++) {
            Real v = fabs(data[i * n_data_cols + j]);
            % if matrix_dim == "rows":
            int relevant_idx = i;
            % else:
            int relevant_idx = j;
            % endif
            // printf("cmp %f %f %i %i \n", v, max_val, relevant_idx, prev[0]);
            if (v > max_val && !in(relevant_idx, prev, n_prev)) {
                // printf("argmax %f %i %i %i \n", v, i, j, n_prev);
                max_idx.row = i;
                max_idx.col = j;
                max_val = v;
            }
        }
    }
    // printf("out argmax %f %i %i %i \n", max_val, max_idx.row, max_idx.col, n_prev);
    return max_idx;
}
%endfor

<%def name="sub_residual(output, rowcol_start, rowcol_end, n_terms, matrix_dim, vec_dim)">
{
    for (int sr_idx = 0; sr_idx < ${n_terms}; sr_idx++) {
        int buffer_ptr = uv_ptrs[uv_ptr0 + sr_idx];
        printf("buffer_ptr: %i k: %i n_terms: %i \n", buffer_ptr, sr_idx, ${n_terms});

        GLOBAL_MEM Real* U_term = &buffer[buffer_ptr];
        GLOBAL_MEM Real* V_term = &buffer[buffer_ptr + n_rows];
        int n_rowcol = (${rowcol_end}) - (${rowcol_start});

        % if matrix_dim == "rows":
            for (int i = 0; i < n_rowcol; i++) {
                Real uv = U_term[i + ${rowcol_start}];
                for (int j = 0; j < n_cols; j++) {
                    Real vv = V_term[j];
                    ${output}[i * n_cols + j] -= uv * vv;
                }
            }
        % else:
            for (int i = 0; i < n_rows; i++) {
                Real uv = U_term[i];
                for (int j = 0; j < n_rowcol; j++) {
                    Real vv = V_term[j + ${rowcol_start}];
                    if (i == 0 && j == 0) {
                        printf("subresid: %f %f\n", uv, vv);
                    }
                    ${output}[i * n_rowcol + j] -= uv * vv;
                }
            }
        % endif
    }
}
</%def>

<%def name="aca(name, evaluator, vec_dim)">

% for matrix_dim in ["rows", "cols"]:
WITHIN_KERNEL 
void calc_${matrix_dim}_${name}(
    GLOBAL_MEM Real* output, 
    int rowcol_start, int rowcol_end,
    GLOBAL_MEM Real* obs_pts, GLOBAL_MEM Real* tris,
    int os, int oe, int ss, int se,
    Real nu)
{
    /*
    * In the calc_rows/cols function, we will calculate a batch of rows or
    * columns corresponding to a particular triangular dislocation element. 
    * In most cases, this will be three rows corresponding to the x/y/z
    * components. But in the case of calculating rows for a strain matrix, we
    * will be calculating six components. See the use of "vec_dim" below to
    * specify the number of rows. 
    * 
    * But, we specify the element in terms of the rowcol_start and rowcol_end.
    * This allows grabbing just a subset of the rows when that is desirable.
    */
    int n_src = se - ss;

    % if matrix_dim == "rows":

    int i = floor(((float)rowcol_start) / ${vec_dim});
    int obs_dim_start = rowcol_start - i * ${vec_dim};
    int obs_dim_end = rowcol_end - i * ${vec_dim};
    int obs_idx = 0;
    // printf("rcs, i, s, e: %i, %i, %i, %i \n", rowcol_start, i, obs_dim_start, obs_dim_end);

    int src_dim_start = 0;
    int src_dim_end = 3;
    int n_output_src = n_src;
    for (int j = ss; j < se; j++) {
        int src_idx = j - ss;

    % else:

    int j = floor(((float)rowcol_start) / 3);
    int src_dim_start = rowcol_start - j * 3;
    int src_dim_end = rowcol_end - j * 3;
    // printf("j, s, e: %i, %i, %i \n", j, src_dim_start, src_dim_end);
    int src_idx = 0;
    int n_output_src = 1;

    int obs_dim_start = 0;
    int obs_dim_end = ${vec_dim};
    for (int i = os; i < oe; i++) {
        int obs_idx = i - os;

    % endif
        Real3 obs;
        // printf("computing ${matrix_dim}_${name} %i %i \n", i, j);
        % for d1 in range(3):
            obs.${comp(d1)} = obs_pts[i * 3 + ${d1}];
        % endfor
        // printf("obs: %f10 %f10 %f10 \n", obs.x, obs.y, obs.z);


        % for d1 in range(3):
            Real3 tri${d1};
            % for d2 in range(3):
                tri${d1}.${comp(d2)} = tris[j * 9 + ${d1} * 3 + ${d2}];
            % endfor
            // printf("tri${d1}: %f10 %f10 %f10 \n", tri${d1}.x, tri${d1}.y, tri${d1}.z);
        % endfor
        // printf("nu: %f \n", nu);

        for (int d_src = src_dim_start; d_src < src_dim_end; d_src++) {
            Real3 slip = make3(0.0, 0.0, 0.0);
            if (d_src == 0) {
                slip.y = 1.0;
            } else if (d_src == 1) {
                slip.z = 1.0;
            } else {
                slip.x = 1.0;
            }

            ${common.setup_tde()}

            ${evaluator()}

            %for d_obs in range(vec_dim):
            {
                if (${d_obs} >= obs_dim_start && ${d_obs} < obs_dim_end) {
                    int idx = (
                        (obs_idx * ${vec_dim} + (${d_obs} - obs_dim_start)) * n_output_src + src_idx
                    ) * (src_dim_end - src_dim_start) + (d_src - src_dim_start);
                    // printf("idx %i %f \n", idx, final.${comp(d_obs)});
                    output[idx] = final.${comp(d_obs)};
                }
            }
            %endfor
        }
    }
}
% endfor

KERNEL
void aca_${name}(
    // out parameters here
    GLOBAL_MEM Real* buffer,
    GLOBAL_MEM int* uv_ptrs, 
    GLOBAL_MEM int* n_terms,
    // mutable workspace parameters
    GLOBAL_MEM int* next_buffer_ptr,
    GLOBAL_MEM Real* fworkspace,
    GLOBAL_MEM int* iworkspace,
    // immutable parameters below here
    GLOBAL_MEM int* uv_ptrs_starts,
    GLOBAL_MEM int* Iref0, GLOBAL_MEM int* Jref0,
    GLOBAL_MEM Real* obs_pts, GLOBAL_MEM Real* tris,
    GLOBAL_MEM int* obs_start, GLOBAL_MEM int* obs_end,
    GLOBAL_MEM int* src_start, GLOBAL_MEM int* src_end,
    Real nu, Real tol, int p_max_iter)
{
    int block_idx = get_global_id(0);
    int os = obs_start[block_idx];
    int oe = obs_end[block_idx];
    int n_obs = oe - os;
    int n_rows = n_obs * ${vec_dim};

    int ss = src_start[block_idx];
    int se = src_end[block_idx];
    int n_src = se - ss;
    int n_cols = n_src * 3;

    int uv_ptr0 = uv_ptrs_starts[block_idx];

    GLOBAL_MEM int* prevIstar = iworkspace;
    GLOBAL_MEM int* prevJstar = &iworkspace[min(n_cols, n_rows) / 2];


    GLOBAL_MEM Real* RIstar = fworkspace;
    GLOBAL_MEM Real* RJstar = &fworkspace[n_cols];

    GLOBAL_MEM Real* RIref = &fworkspace[n_cols + n_rows];
    printf("fworkspace: %i %i %i %i \n", 0, n_cols, n_cols + n_rows, n_cols + n_rows + 3 * n_cols);
    GLOBAL_MEM Real* RJref = &fworkspace[n_cols + n_rows + 3 * n_cols];

    int Iref = Iref0[block_idx];
    Iref -= Iref % 3;
    int Jref = Jref0[block_idx];
    Jref -= Jref % 3;

    calc_rows_${name}(
        RIref, Iref, Iref + ${vec_dim},
        obs_pts, tris, os, oe, ss, se, nu
    );

    calc_cols_${name}(
        RJref, Jref, Jref + 3, 
        obs_pts, tris, os, oe, ss, se, nu
    );
    // printf("%f\n", RIref[0]);

    int max_iter = min(p_max_iter, min(n_rows / 2, n_cols / 2));
    // TODO:
    // TODO:
    // TODO:
    // TODO:
    // max_iter = 10;


    Real frob_est = 0;
    int k = 0;
    for (; k < max_iter; k++) {
        printf("\n\nstart iteration %i\n", k);
        for (int i = 0; i < 5; i++) {
            printf("RIref[%i] = %f\n", i, RIref[i]);
        }
        for (int j = 0; j < 5; j++) {
            printf("RJref[%i] = %f\n", j, RJref[j]);
        }

        struct MatrixIndex Istar_entry = argmax_abs_not_in_list_rows(RJref, n_rows, 3, prevIstar, k);
        struct MatrixIndex Jstar_entry = argmax_abs_not_in_list_cols(RIref, ${vec_dim}, n_cols, prevJstar, k);
        int Istar = Istar_entry.row;
        int Jstar = Jstar_entry.col;

        Real Istar_val = fabs(RJref[Istar_entry.row * 3 + Istar_entry.col]);
        Real Jstar_val = fabs(RIref[Jstar_entry.row * n_cols + Jstar_entry.col]);
        printf("pivot guess %i %i %f %f \n", Istar, Jstar, Istar_val, Jstar_val);
        if (Istar_val > Jstar_val) {
            calc_rows_${name}(
                RIstar, Istar, Istar + 1,
                obs_pts, tris, os, oe, ss, se, nu
            );
            ${sub_residual("RIstar", "Istar", "Istar + 1", "k", "rows", vec_dim)}

            Jstar_entry = argmax_abs_not_in_list_cols(RIstar, 1, n_cols, prevJstar, k);
            printf("Jstar_entry: %i %i \n", Jstar_entry.row, Jstar_entry.col);
            Jstar = Jstar_entry.col;

            calc_cols_${name}(
                RJstar, Jstar, Jstar + 1,
                obs_pts, tris, os, oe, ss, se, nu
            );
            ${sub_residual("RJstar", "Jstar", "Jstar + 1", "k", "cols", vec_dim)}
        } else {
            calc_cols_${name}(
                RJstar, Jstar, Jstar + 1,
                obs_pts, tris, os, oe, ss, se, nu
            );
            ${sub_residual("RJstar", "Jstar", "Jstar + 1", "k", "cols", vec_dim)}

            for (int j = 0; j < 5; j++) {
                printf("RJstar[%i] = %f\n", j, RJstar[j]);
            }


            Istar_entry = argmax_abs_not_in_list_rows(RJstar, n_rows, 1, prevIstar, k);
            printf("Istar_entry: %i %i \n", Istar_entry.row, Istar_entry.col);
            Istar = Istar_entry.row;

            calc_rows_${name}(
                RIstar, Istar, Istar + 1,
                obs_pts, tris, os, oe, ss, se, nu
            );
            ${sub_residual("RIstar", "Istar", "Istar + 1", "k", "rows", vec_dim)}
        }
        printf("true pivot: %i %i \n", Istar, Jstar);
        printf("diagonal %f \n", RIstar[Jstar]);

        prevIstar[k] = Istar;
        prevJstar[k] = Jstar;

        // claim a block of space for the first U and first V vectors and collect
        // the corresponding Real* pointers
        int next_buffer_u_ptr = buffer_alloc(next_buffer_ptr, n_rows + n_cols);
        int next_buffer_v_ptr = next_buffer_u_ptr + n_rows;
        GLOBAL_MEM Real* next_buffer_u = &buffer[next_buffer_u_ptr];
        GLOBAL_MEM Real* next_buffer_v = &buffer[next_buffer_v_ptr];

        // Assign our uv_ptr to point to the u,v buffer location.
        uv_ptrs[uv_ptr0 + k] = next_buffer_u_ptr;

        Real v2 = 0;
        for (int i = 0; i < n_rows; i++) {
            next_buffer_v[i] = RIstar[i] / RIstar[Jstar];
            if (i < 5) {
                printf("v[%i] = %f\n", i, next_buffer_v[i]);
            }
            v2 += next_buffer_v[i] * next_buffer_v[i];
        }

        Real u2 = 0;
        for (int j = 0; j < n_cols; j++) {
            next_buffer_u[j] = RJstar[j];
            if (j < 5) {
                printf("u[%i] = %f\n", j, next_buffer_u[j]);
            }
            u2 += next_buffer_u[j] * next_buffer_u[j];
        }

        Real step_size = sqrt(u2 * v2);

        printf("step_size %f \n", step_size);
        frob_est += step_size;
        printf("frob_est: %f \n", frob_est);

        if (step_size < tol) {
            break;
        }

        if (k == max_iter - 1) {
            break;
        }

        if (Iref <= Istar && Istar < Iref + ${vec_dim}) {
            while (true) {
                Iref = (Iref + ${vec_dim}) % n_rows;
                Iref -= Iref % ${vec_dim};
                if (!in(Iref, prevIstar, k + 1)) {
                    printf("new Iref: %i \n", Iref);
                    break; 
                }
            }
            calc_rows_${name}(
                RIref, Iref, Iref + ${vec_dim},
                obs_pts, tris, os, oe, ss, se, nu
            );
            ${sub_residual("RIref", "Iref", "Iref + " + str(vec_dim), "k + 1", "rows", vec_dim)}
        } else {
            for (int i = 0; i < ${vec_dim}; i++) {
                for (int j = 0; j < n_cols; j++) {
                    RIref[i * n_cols + j] -= next_buffer_u[i + Iref] * next_buffer_v[j];
                }
            }
        }

        if (Jref <= Jstar && Jstar < Jref + ${vec_dim}) {
            while (true) {
                Jref = (Jref + 3) % n_rows;
                Jref -= Jref % 3;
                if (!in(Jref, prevJstar, k + 1)) {
                    printf("new Jref: %i \n", Jref);
                    break; 
                }
            }
            calc_cols_${name}(
                RJref, Jref, Jref + 3, 
                obs_pts, tris, os, oe, ss, se, nu
            );
            ${sub_residual("RJref", "Jref", "Jref + 3", "k + 1", "cols", vec_dim)}
        } else {
            for (int i = 0; i < n_rows; i++) {
                for (int j = 0; j < 3; j++) {
                    RJref[i * 3 + j] -= next_buffer_u[i] * next_buffer_v[j + Jref];
                }
            }
        }
    }

    n_terms[block_idx] = k + 1;
}
</%def>

//TODO: 
//TODO: 
//TODO: 
//TODO: 
//TODO: add strain back in
${aca("disp", common.disp, 3)}
