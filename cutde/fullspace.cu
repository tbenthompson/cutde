<%namespace module="cutde.mako_helpers" import="*"/>
<%namespace name="common" file="common.cu"/>

${common.defs()}

<%def name="tde(name, evaluator, vec_dim)">
KERNEL
void ${name}_fullspace(GLOBAL_MEM Real* results, int n_pairs, 
    GLOBAL_MEM Real* obs_pts, GLOBAL_MEM Real* tris,
    GLOBAL_MEM Real* slips, Real nu)
{
    int i = get_global_id(0);
    if (i >= n_pairs) {
        return;
    }
    Real3 obs;
    % for d1 in range(3):
        Real3 tri${d1};
        obs.${comp(d1)} = obs_pts[i * 3 + ${d1}];
        % for d2 in range(3):
            tri${d1}.${comp(d2)} = tris[i * 9 + ${d1} * 3 + ${d2}];
        % endfor
    % endfor

    Real3 slip = make3(
        slips[i * 3 + 2],
        slips[i * 3 + 0],
        slips[i * 3 + 1]
    );

    ${common.setup_tde()}

    ${evaluator()}

    %for d in range(vec_dim):
        results[i * ${vec_dim} + ${d}] = final.${comp(d)};
    %endfor
}
</%def>

<%def name="tde_matrix(name, evaluator, vec_dim)">
KERNEL
void ${name}_fullspace_matrix(GLOBAL_MEM Real* results, 
    int n_obs, int n_src,
    GLOBAL_MEM Real* obs_pts, GLOBAL_MEM Real* tris,
    Real nu)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= n_obs) {
        return;
    }

    if (j >= n_src) {
        return;
    }

    Real3 obs;
    % for d1 in range(3):
        Real3 tri${d1};
        obs.${comp(d1)} = obs_pts[i * 3 + ${d1}];
        % for d2 in range(3):
            tri${d1}.${comp(d2)} = tris[j * 9 + ${d1} * 3 + ${d2}];
        % endfor
    % endfor

    % for d_src in range(3):
    {
        Real3 slip = make3(0.0, 0.0, 0.0);
        slip.${comp([1,2,0][d_src])} = 1.0;

        ${common.setup_tde()}

        ${evaluator()}

        %for d_obs in range(vec_dim):
        {
            int idx = ((i * ${vec_dim} + ${d_obs}) * n_src + j) * 3 + ${d_src};
            results[idx] = final.${comp(d_obs)};
        }
        %endfor
    }
    % endfor
}
</%def>

<%def name="tde_free(name, evaluator, vec_dim)">
KERNEL
void ${name}_fullspace_free(GLOBAL_MEM Real* results, 
    int n_obs, int n_src,
    GLOBAL_MEM Real* obs_pts, GLOBAL_MEM Real* tris,
    GLOBAL_MEM Real* slips,
    Real nu)
{
    int i = get_global_id(0);
    int group_id = get_local_id(0);

    %for d_obs in range(vec_dim):
        Real sum${d_obs} = 0.0;
        Real kahanc${d_obs} = 0.0;
    %endfor

    Real3 obs;
    if (i < n_obs) {
        % for d1 in range(3):
            obs.${comp(d1)} = obs_pts[i * 3 + ${d1}];
        % endfor
    }

    % for d1 in range(3):
        LOCAL_MEM Real3 sh_tri${d1}[${block_size}];
    % endfor
    LOCAL_MEM Real3 sh_slips[${block_size}];

    for (int block_start = 0; block_start < n_src; block_start += ${block_size}) {
        int j = block_start + group_id;
        if (j < n_src) {
            % for d1 in range(3):
                % for d2 in range(3):
                    sh_tri${d1}[group_id].${comp(d2)} = tris[j * 9 + ${d1} * 3 + ${d2}];
                % endfor
                sh_slips[group_id].${comp([1,2,0][d1])} = slips[j * 3 + ${d1}];
            % endfor
        }

        LOCAL_BARRIER;

        if (i < n_obs) {
            int block_end = min(n_src, block_start + ${block_size});
            int block_length = block_end - block_start;
            for (int block_idx = 0; block_idx < block_length; block_idx++) {
                % for d1 in range(3):
                    Real3 tri${d1} = sh_tri${d1}[block_idx];
                % endfor

                Real3 slip = sh_slips[block_idx];

                ${common.setup_tde()}

                ${evaluator()}

                %for d_obs in range(vec_dim):
                {
                    Real input = final.${comp(d_obs)};
                    Real y = input - kahanc${d_obs};
                    Real t = sum${d_obs} + y;
                    kahanc${d_obs} = (t - sum${d_obs}) - y;
                    sum${d_obs} = t;
                }
                %endfor
            }
        }
    }

    if (i < n_obs) {
        %for d_obs in range(vec_dim):
            results[i * ${vec_dim} + ${d_obs}] = sum${d_obs};
        %endfor
    }
}
</%def>

<%def name="tde_blocks(name, evaluator, vec_dim)">
KERNEL
void ${name}_fullspace_blocks(GLOBAL_MEM Real* results, 
    GLOBAL_MEM Real* obs_pts, GLOBAL_MEM Real* tris,
    GLOBAL_MEM int* obs_start, GLOBAL_MEM int* obs_end,
    GLOBAL_MEM int* src_start, GLOBAL_MEM int* src_end,
    GLOBAL_MEM int* block_start,
    Real nu)
{
    int block_idx = get_global_id(0);
    int os = obs_start[block_idx];
    int oe = obs_end[block_idx];
    int n_obs = oe - os;

    int ss = src_start[block_idx];
    int se = src_end[block_idx];
    int n_src = se - ss;

    int bs = block_start[block_idx];

    for (int i = os; i < oe; i++) {
        int obs_idx = i - os;

        Real3 obs;
        % for d1 in range(3):
            obs.${comp(d1)} = obs_pts[i * 3 + ${d1}];
        % endfor

        for (int j = ss; j < se; j++) {
            int src_idx = j - ss;

            % for d1 in range(3):
                Real3 tri${d1};
                % for d2 in range(3):
                    tri${d1}.${comp(d2)} = tris[j * 9 + ${d1} * 3 + ${d2}];
                % endfor
            % endfor
            % for d_src in range(3):
            {
                Real3 slip = make3(0.0, 0.0, 0.0);
                slip.${comp([1,2,0][d_src])} = 1.0;

                ${common.setup_tde()}

                ${evaluator()}

                %for d_obs in range(vec_dim):
                {
                    int idx = bs + (
                        (obs_idx * ${vec_dim} + ${d_obs}) * n_src + src_idx
                    ) * 3 + ${d_src};
                    results[idx] = final.${comp(d_obs)};
                }
                %endfor
            }
            % endfor
        }
    }
}
</%def>

${tde("disp", common.disp, 3)}
${tde("strain", common.strain, 6)}
${tde_matrix("disp", common.disp, 3)}
${tde_matrix("strain", common.strain, 6)}
${tde_free("disp", common.disp, 3)}
${tde_free("strain", common.strain, 6)}
${tde_blocks("strain", common.strain, 6)}
${tde_blocks("disp", common.disp, 3)}
